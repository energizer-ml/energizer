from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energizer._mlx import mx
from .tracer import IRNode


@dataclass
class BouncerConfig:
    prefer_fp16: bool = True
    fuse_linear_bias: bool = True
    canonicalize_relu: bool = True
    fuse_silu: bool = True
    fuse_attention: bool = True


class Bouncer:
    """
    Rewrites the traced IR into ANE-friendlier CoreML ops.

    This pass does not guarantee ANE placement. CoreML still decides final device
    scheduling at runtime, but fused ops, static constants and fp16 hints improve
    the odds of running larger portions of the graph on the ANE.
    """

    ANE_FRIENDLY_OPS = {
        "Linear",
        "ReLU",
        "Sigmoid",
        "Softmax",
        "GELU",
        "SiLU",
        "LayerNorm",
        "ScaledDotProductAttention",
        "Mul",
        "Add",
        "Transpose",
    }

    def __init__(self, config: BouncerConfig | None = None):
        self.config = config or BouncerConfig()

    def bounce(self, nodes: list[IRNode]) -> list[IRNode]:
        rewritten: list[IRNode] = []
        consumed: set[int] = set()
        aliases: dict[IRNode, IRNode] = {}

        for idx, node in enumerate(nodes):
            if idx in consumed:
                continue
            node = self._resolve_node(node, aliases)

            if self.config.fuse_linear_bias:
                linear = self._try_fuse_linear(nodes, idx)
                if linear is not None:
                    linear = self._resolve_node(linear, aliases)
                    rewritten.append(linear)
                    aliases[nodes[idx]] = linear
                    aliases[nodes[idx + 1]] = linear
                    consumed.add(idx)
                    consumed.add(idx + 1)
                    continue

            if self.config.canonicalize_relu:
                relu = self._try_canonicalize_relu(node)
                if relu is not None:
                    aliases[nodes[idx]] = relu
                    rewritten.append(relu)
                    continue

            if self.config.fuse_silu:
                silu = self._try_fuse_silu(node)
                if silu is not None:
                    aliases[nodes[idx]] = silu
                    rewritten.append(silu)
                    continue

            if self.config.fuse_attention:
                attention = self._try_fuse_attention(node)
                if attention is not None:
                    aliases[nodes[idx]] = attention
                    rewritten.append(attention)
                    continue

            annotated = self._annotate(node)
            aliases[nodes[idx]] = annotated
            rewritten.append(annotated)

        return self._eliminate_dead_code(rewritten)

    def _resolve_value(self, value: Any, aliases: dict[IRNode, IRNode]) -> Any:
        while isinstance(value, IRNode) and value in aliases:
            next_value = aliases[value]
            if next_value is value:
                break
            value = next_value
        return value

    def _resolve_node(self, node: IRNode, aliases: dict[IRNode, IRNode]) -> IRNode:
        resolved_inputs = [self._resolve_value(inp, aliases) for inp in node.inputs]
        if resolved_inputs == node.inputs:
            return node
        return IRNode(
            node.op,
            resolved_inputs,
            node.output_shape,
            node.output_dtype,
            dict(node.attrs),
        )

    def _annotate(self, node: IRNode) -> IRNode:
        attrs = dict(node.attrs)
        attrs.setdefault("ane_preferred", node.op in self.ANE_FRIENDLY_OPS)
        attrs.setdefault("precision", "fp16" if self.config.prefer_fp16 else "fp32")
        return IRNode(node.op, list(node.inputs), node.output_shape, node.output_dtype, attrs)

    def _eliminate_dead_code(self, nodes: list[IRNode]) -> list[IRNode]:
        if not nodes:
            return nodes

        live: set[int] = set()

        def mark(node: IRNode):
            node_id = id(node)
            if node_id in live:
                return
            live.add(node_id)
            for inp in node.inputs:
                if isinstance(inp, IRNode):
                    mark(inp)

        mark(nodes[-1])
        return [node for node in nodes if id(node) in live]

    def _try_fuse_linear(self, nodes: list[IRNode], idx: int) -> IRNode | None:
        if idx + 1 >= len(nodes):
            return None

        matmul = nodes[idx]
        add = nodes[idx + 1]

        if matmul.op != "MatMul" or add.op != "Add":
            return None

        left, right = add.inputs
        bias = None
        if left is matmul and self._is_const_bias(right, add.output_shape):
            bias = right
        elif right is matmul and self._is_const_bias(left, add.output_shape):
            bias = left
        if bias is None:
            return None

        if len(matmul.inputs) != 2:
            return None
        x, weight = matmul.inputs
        if not self._is_const(weight):
            return None

        attrs = {
            "ane_preferred": True,
            "precision": "fp16" if self.config.prefer_fp16 else "fp32",
            "fused_from": ["MatMul", "Add"],
        }
        return IRNode(
            "Linear",
            [x, weight, bias],
            add.output_shape,
            add.output_dtype,
            attrs,
        )

    def _try_canonicalize_relu(self, node: IRNode) -> IRNode | None:
        if node.op != "Maximum" or len(node.inputs) != 2:
            return None

        left, right = node.inputs
        if self._is_zero(left):
            x = right
        elif self._is_zero(right):
            x = left
        else:
            return None

        attrs = {
            "ane_preferred": True,
            "precision": "fp16" if self.config.prefer_fp16 else "fp32",
            "fused_from": ["Maximum"],
        }
        return IRNode("ReLU", [x], node.output_shape, node.output_dtype, attrs)

    def _try_fuse_silu(self, node: IRNode) -> IRNode | None:
        if node.op != "Mul" or len(node.inputs) != 2:
            return None

        left, right = node.inputs
        if isinstance(left, IRNode) and left.op == "Sigmoid" and self._same_source(
            left.inputs[0], right
        ):
            x = right
        elif isinstance(right, IRNode) and right.op == "Sigmoid" and self._same_source(
            right.inputs[0], left
        ):
            x = left
        else:
            return None

        attrs = {
            "ane_preferred": True,
            "precision": "fp16" if self.config.prefer_fp16 else "fp32",
            "fused_from": ["Sigmoid", "Mul"],
        }
        return IRNode("SiLU", [x], node.output_shape, node.output_dtype, attrs)

    def _try_fuse_attention(self, node: IRNode) -> IRNode | None:
        if node.op != "MatMul" or len(node.inputs) != 2:
            return None

        attn_weights, value = node.inputs
        if not isinstance(attn_weights, IRNode) or attn_weights.op != "Softmax":
            return None
        if len(attn_weights.inputs) != 1:
            return None

        scale = np.float32(1.0)
        attn_mask = None
        scores = attn_weights.inputs[0]

        if isinstance(scores, IRNode) and scores.op == "Add" and len(scores.inputs) == 2:
            left, right = scores.inputs
            if self._looks_like_attention_scores(left):
                scores, attn_mask = left, right
            elif self._looks_like_attention_scores(right):
                scores, attn_mask = right, left

        if isinstance(scores, IRNode) and scores.op in {"Mul", "Div"} and len(scores.inputs) == 2:
            scale_op = scores.op
            left, right = scores.inputs
            if self._is_const(left):
                const, scores = left, right
            elif self._is_const(right):
                const, scores = right, left
            else:
                const = None

            if const is not None:
                const_scalar = self._to_scalar(const)
                if const_scalar is not None:
                    if scale_op == "Mul":
                        scale = np.float32(const_scalar)
                    else:
                        scale = np.float32(1.0 / const_scalar)

        if not isinstance(scores, IRNode) or scores.op != "MatMul" or len(scores.inputs) != 2:
            return None

        query, key = scores.inputs
        if not isinstance(key, IRNode) or key.op != "Transpose" or len(key.inputs) != 1:
            return None

        attrs = {
            "ane_preferred": True,
            "precision": "fp16" if self.config.prefer_fp16 else "fp32",
            "fused_from": ["MatMul", "Mul", "Softmax", "MatMul"],
            "scale": float(scale),
        }
        return IRNode(
            "ScaledDotProductAttention",
            [query, key.inputs[0], value, scale, attn_mask],
            node.output_shape,
            node.output_dtype,
            attrs,
        )

    def _looks_like_attention_scores(self, value: Any) -> bool:
        if not isinstance(value, IRNode):
            return False
        if value.op == "MatMul":
            return True
        if value.op in {"Mul", "Div"} and len(value.inputs) == 2:
            return any(
                isinstance(inp, IRNode) and inp.op == "MatMul" for inp in value.inputs
            )
        return False

    def _same_source(self, a: Any, b: Any) -> bool:
        if a is b:
            return True
        if isinstance(a, IRNode) or isinstance(b, IRNode):
            return False
        return id(a) == id(b)

    def _is_const_bias(self, value: Any, output_shape: tuple) -> bool:
        if not self._is_const(value):
            return False
        if len(output_shape) == 0:
            return True
        shape = self._shape_of(value)
        if shape == ():
            return True
        return len(shape) == 1 and shape[0] == output_shape[-1]

    def _is_const(self, value: Any) -> bool:
        if isinstance(value, IRNode):
            return False
        if isinstance(value, (int, float, np.number, np.ndarray)):
            return True
        if mx is not None and isinstance(value, mx.array):
            return True
        return hasattr(value, "numpy") or hasattr(value, "data")

    def _shape_of(self, value: Any) -> tuple:
        if isinstance(value, (int, float, np.number)):
            return ()
        if isinstance(value, np.ndarray):
            return value.shape
        if mx is not None and isinstance(value, mx.array):
            return value.shape
        if hasattr(value, "shape"):
            return value.shape
        if hasattr(value, "data") and hasattr(value.data, "shape"):
            return value.data.shape
        return ()

    def _is_zero(self, value: Any) -> bool:
        if isinstance(value, IRNode):
            return False
        if isinstance(value, (int, float, np.number)):
            return float(value) == 0.0
        if isinstance(value, np.ndarray):
            return value.shape == () and float(value) == 0.0
        if mx is not None and isinstance(value, mx.array):
            return value.shape == () and float(np.array(value)) == 0.0
        if hasattr(value, "numpy"):
            arr = value.numpy()
            return arr.shape == () and float(arr) == 0.0
        if hasattr(value, "data"):
            data = value.data
            if isinstance(data, np.ndarray):
                return data.shape == () and float(data) == 0.0
            if mx is not None and isinstance(data, mx.array):
                return data.shape == () and float(np.array(data)) == 0.0
        return False

    def _to_scalar(self, value: Any) -> float | None:
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return float(value)
            return None
        if hasattr(value, "numpy"):
            arr = value.numpy()
            if getattr(arr, "shape", None) == ():
                return float(arr)
            return None
        if hasattr(value, "data"):
            data = value.data
            if isinstance(data, np.ndarray) and data.shape == ():
                return float(data)
            if mx is not None and isinstance(data, mx.array):
                arr = np.array(data)
                if arr.shape == ():
                    return float(arr)
        return None
