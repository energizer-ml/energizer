"""
Autograd engine for Energizer.

Responsibilities:
    - Track the computation graph (nodes = Tensors, edges = Functions)
    - Accumulate gradients correctly (handles shared weights / reuse)
    - Traverse the graph in topological order for backward()
    - Delegate actual math to Backend
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from energizer.backend import backend, MLX_AVAILABLE

# ── Function base ────────────────────────────────────────────────────────────


class Function:
    """
    Base class for all differentiable operations.

    Subclass this and implement:
        forward(ctx, *args)   → raw array result
        backward(ctx, grad)   → tuple of raw array grads (one per input)

    'ctx' is a Context object used to save data needed for backward.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        """
        Main entry point. Call this instead of forward() directly.
        Builds the graph node and runs forward.
        """
        first_tensor = next(t for t in args if isinstance(t, Tensor))

        from energizer.backpack.compiler.tracer import Tracer, TraceData, IRNode

        if Tracer.is_tracing():
            tracer = Tracer.get()
            out_shape = Tracer.infer_shape(cls.__name__, *args)

            inputs = []
            for a in args:
                if isinstance(a, Tensor):
                    if hasattr(a, "_ir_node") and a._ir_node is not None:
                        inputs.append(a._ir_node)
                    else:
                        inputs.append(a)
                else:
                    inputs.append(a)

            node = IRNode(cls.__name__, inputs, out_shape, "float32")
            tracer.nodes.append(node)
            out = Tensor(TraceData(out_shape, "float32"), device=first_tensor.device)
            out._ir_node = node
            return out

        ctx = Context(device=first_tensor.device)

        raw_inputs = [t.data if isinstance(t, Tensor) else t for t in args]
        raw_out = cls.forward(ctx, *raw_inputs)

        requires_grad = any(getattr(t, "requires_grad", False) for t in args)
        result = Tensor(
            raw_out, device=first_tensor.device, requires_grad=requires_grad
        )

        if requires_grad:
            result._node = GraphNode(
                function=cls,
                ctx=ctx,
                inputs=list(args),
                parents=[t for t in args if getattr(t, "requires_grad", False)],
            )

        return result


# ── Context ──────────────────────────────────────────────────────────────────


class Context:
    """
    Passed to forward() and backward().
    Stores intermediate values needed for the backward pass.
    Stores raw arrays (np/mx), NOT Tensor objects — avoids circular refs.
    """

    def __init__(self, device: str):
        self.device = device
        self._saved = []
        self._cache = {}  # for arbitrary metadata (e.g. kernel size)

    def save_for_backward(self, *arrays):
        """Save raw arrays to use in backward()."""
        self._saved = list(arrays)

    @property
    def saved_tensors(self):
        return self._saved

    def __setattr__(self, key, value):
        if key.startswith("_") or key == "device":
            super().__setattr__(key, value)
        else:
            # ctx.stride = 2  → stored in _cache
            self._cache[key] = value

    def __getattr__(self, key):
        try:
            return self.__dict__["_cache"][key]
        except KeyError:
            raise AttributeError(f"Context has no attribute '{key}'")


# ── Graph node ───────────────────────────────────────────────────────────────


class GraphNode:
    """
    One node in the computation graph.
    Attached to a Tensor as tensor._node.

    - inputs:  ALL tensors fed into the op (needed to align backward() output)
    - parents: only those with requires_grad=True (used for topo traversal)
    """

    __slots__ = ("function", "ctx", "inputs", "parents")

    def __init__(self, function: type, ctx: Context, inputs: list, parents: list):
        self.function = function
        self.ctx = ctx
        self.inputs = inputs  # ALL inputs in forward order
        self.parents = parents  # subset: only requires_grad=True


# ── Tensor ───────────────────────────────────────────────────────────────────


class Tensor:
    """
    Core data structure. Wraps a raw np.ndarray or mx.array.
    Carries graph metadata for autograd.
    """

    def __init__(self, data, device: str = "cpu", requires_grad: bool = False):
        if isinstance(data, Tensor):
            if device == "cpu" and data.device != "cpu":
                device = data.device
            data = data.data

        # Always convert to the correct backend array type for this device.
        # This handles: plain lists, scalars, np.ndarray on gpu, mx.array on cpu.
        from energizer.backpack.compiler.tracer import TraceData

        if isinstance(data, TraceData):
            pass
        elif device == "gpu" and MLX_AVAILABLE:
            import mlx.core as mx_local

            if not isinstance(data, mx_local.array):
                data = mx_local.array(data)
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            elif data.dtype != np.float32:
                data = data.astype(np.float32)

        self.data = data
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None  # filled by backward()
        self._node: Optional[GraphNode] = None

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, grad: Optional[Tensor] = None):
        """
        Trigger reverse-mode autodiff from this tensor.
        Call on a scalar loss: loss.backward()
        """
        if not self.requires_grad:
            raise RuntimeError("Called backward() on a tensor with requires_grad=False")

        # Seed gradient: default to 1.0 for scalar loss
        if grad is None:
            if self.data.shape != () and self.data.size != 1:
                raise RuntimeError(
                    "backward() without a gradient argument is only supported "
                    "for scalar outputs. Pass grad= for non-scalar tensors."
                )
            seed = backend.ones((), device=self.device)
        else:
            seed = grad.data

        # ── Topological sort ─────────────────────────────────────────────────
        # Collect all nodes that have a _node (non-leaf intermediate tensors).
        # Leaf tensors (no _node) are NOT in order, but their parents are
        # referenced from nodes that ARE in order — we write their grads
        # separately at the end.
        order = []
        visited = set()

        def topo(t: Tensor):
            if id(t) in visited or t._node is None:
                return
            visited.add(id(t))
            for parent in t._node.parents:
                topo(parent)
            order.append(t)

        topo(self)

        # Also track every tensor we encounter (including leaves)
        # so we can write grads onto them at the end.
        all_tensors: dict[int, Tensor] = {}

        def collect(t):
            if not isinstance(t, Tensor):
                return
            if id(t) in all_tensors:
                return
            all_tensors[id(t)] = t
            if getattr(t, "_node", None) is not None:
                for inp in t._node.inputs:
                    collect(inp)

        collect(self)

        # ── Reverse pass ─────────────────────────────────────────────────────
        grads: dict[int, object] = {id(self): seed}

        for t in reversed(order):
            if t._node is None:
                continue

            g = grads.get(id(t))
            if g is None:
                continue

            # Call the function's backward with raw array grad
            input_grads = t._node.function.backward(t._node.ctx, g)

            # backward() may return a single array or a tuple
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            # Zip over ALL inputs (not just parents) so positions align.
            for inp, pg in zip(t._node.inputs, input_grads):
                if pg is None or not getattr(inp, "requires_grad", False):
                    continue

                # Ensure gradient has the same shape as the input tensor
                # (reduces broadcasted dimensions).
                pg = _unbroadcast_grad(pg, inp.shape, inp.device)

                pid = id(inp)
                if pid in grads:
                    grads[pid] = grads[pid] + pg
                else:
                    grads[pid] = pg

        # ── Write final grads onto ALL requires_grad tensors ─────────────────
        for tid, t in all_tensors.items():
            if not t.requires_grad:
                continue
            if tid not in grads:
                continue
            g = grads[tid]
            if t.grad is None:
                t.grad = Tensor(g, device=t.device)
            else:
                # Accumulate across multiple backward() calls (e.g. called twice)
                t.grad = Tensor(t.grad.data + g, device=t.device)

    def zero_grad(self):
        self.grad = None

    # ── Device transfer ───────────────────────────────────────────────────────

    def to(self, device: str) -> Tensor:
        if device == self.device:
            return self
        new_data = backend.transfer(self.data, self.device, device)
        return Tensor(new_data, device=device, requires_grad=self.requires_grad)

    # ── Operator overloads (delegate to Function subclasses) ─────────────────

    def __add__(self, other):
        return Add.apply(self, _wrap(other, self))

    def __sub__(self, other):
        return Sub.apply(self, _wrap(other, self))

    def __mul__(self, other):
        return Mul.apply(self, _wrap(other, self))

    def __truediv__(self, other):
        return Div.apply(self, _wrap(other, self))

    def __matmul__(self, other):
        return MatMul.apply(self, _wrap(other, self))

    def __neg__(self):
        return Neg.apply(self)

    def __pow__(self, exp):
        return Pow.apply(self, _wrap(exp, self))

    def sum(self, axis=None):
        return Sum.apply(self) if axis is None else SumAxis.apply(self, axis)

    def mean(self, axis=None):
        return Mean.apply(self)

    def __repr__(self):
        return f"Tensor({self.data}, device='{self.device}', requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return Transpose.apply(self)

    def item(self):
        """Extract a Python scalar from a single-element tensor."""
        data = self.data
        # MLX arrays need to be evaluated first
        if _is_mlx(data):
            return data.item()
        arr = np.asarray(data)
        if arr.size != 1:
            raise ValueError(
                f"item() only works on single-element tensors, got shape {arr.shape}"
            )
        return float(arr.flat[0])

    def tanh(self):
        return Tanh.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def softmax(self):
        return Softmax.apply(self)

    def numpy(self):
        return backend.to_numpy(self.data, self.device)

    def mlx(self):
        if self.device == "gpu":
            return self.data
        return backend.transfer(self.data, self.device, "gpu")

    def cpu(self):
        return self.to("cpu")

    def gpu(self):
        return self.to("gpu")

    def copy(self):
        d = getattr(self.data, "copy", lambda: self.data)()
        return Tensor(d, device=self.device, requires_grad=self.requires_grad)

    @property
    def size(self):
        return self.data.shape

    @staticmethod
    def randn(*args, device="cpu", **kwargs):
        return Tensor(
            backend.randn(*args, device=device), requires_grad=False, device=device
        )

    @staticmethod
    def zeros(shape, device="cpu"):
        return Tensor(
            backend.zeros(shape, device=device), requires_grad=False, device=device
        )

    @staticmethod
    def ones(shape, device="cpu"):
        return Tensor(
            backend.ones(shape, device=device), requires_grad=False, device=device
        )

    def reshape(self, shape):
        return Reshape.apply(self, shape)

    def view(self, shape):
        return self.reshape(shape)

    def squeeze(self, axis=None):
        return Squeeze.apply(self, axis)

    def clamp(self, min_val, max_val):
        return Clamp.apply(self, min_val, max_val)

    def minimum(self, other):
        return Minimum.apply(self, _wrap(other, self))

    def maximum(self, other):
        return Maximum.apply(self, _wrap(other, self))

    def __getitem__(self, key):
        return GetItem.apply(self, key)


# ── Built-in Functions ───────────────────────────────────────────────────────
# Each one calls backend.* so the math is always dispatched correctly.


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * b, grad * a


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad / b, -grad * a / (b**2)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return backend.matmul(a, b, ctx.device)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = backend.matmul(grad, backend.transpose(b, ctx.device), ctx.device)
        grad_b = backend.matmul(backend.transpose(a, ctx.device), grad, ctx.device)
        return grad_a, grad_b


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad):
        return (-grad,)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exp):
        ctx.save_for_backward(a, exp)
        return a**exp

    @staticmethod
    def backward(ctx, grad):
        a, exp = ctx.saved_tensors
        return (exp * a ** (exp - 1) * grad, None)  # None = no grad for exponent


class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.sum(a, device=ctx.device)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        return (backend.ones(a.shape, ctx.device) * grad,)


class SumAxis(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.save_for_backward(a.shape, axis)
        return backend.sum(a, axis=axis, keepdims=False, device=ctx.device)

    @staticmethod
    def backward(ctx, grad):
        orig_shape, axis = ctx.saved_tensors
        import numpy as np

        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            grad_expanded = grad
            for ax in sorted(axis):
                if hasattr(grad_expanded, "shape"):
                    pass
                grad_expanded = backend.lib(ctx.device).expand_dims(
                    grad_expanded, axis=ax
                )
        else:
            grad_expanded = grad

        return (backend.ones(orig_shape, ctx.device) * grad_expanded, None)


class Mean(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.mean(a, device=ctx.device)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        n = a.size if hasattr(a, "size") else np.prod(a.shape)
        return (backend.ones(a.shape, ctx.device) * grad / n,)


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        return backend.transpose(a, ctx.device)

    @staticmethod
    def backward(ctx, grad):
        return (backend.transpose(grad, ctx.device),)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = backend.exp(a, ctx.device)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors
        return (out * grad,)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.log(a, ctx.device)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        return (grad / a,)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, tuple_shape):
        ctx.save_for_backward(a.shape)
        if hasattr(a, "reshape"):
            return a.reshape(tuple_shape)
        import numpy as np

        return np.reshape(a, tuple_shape)

    @staticmethod
    def backward(ctx, grad):
        (orig_shape,) = ctx.saved_tensors
        if hasattr(grad, "reshape"):
            return (grad.reshape(orig_shape), None)
        import numpy as np

        return (np.reshape(grad, orig_shape), None)


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.save_for_backward(a.shape, axis)
        if hasattr(a, "squeeze"):
            return a.squeeze() if axis is None else a.squeeze(axis=axis)
        import numpy as np

        return np.squeeze(a) if axis is None else np.squeeze(a, axis=axis)

    @staticmethod
    def backward(ctx, grad):
        orig_shape, axis = ctx.saved_tensors
        if hasattr(grad, "reshape"):
            return (grad.reshape(orig_shape), None)
        import numpy as np

        return (np.reshape(grad, orig_shape), None)


class Clamp(Function):
    @staticmethod
    def forward(ctx, a, min_val, max_val):
        ctx.save_for_backward(a, min_val, max_val)
        return backend.clip(a, min_val, max_val, ctx.device)

    @staticmethod
    def backward(ctx, grad):
        a, min_val, max_val = ctx.saved_tensors
        mask = (a >= min_val) & (a <= max_val)
        return (grad * mask, None, None)


class Minimum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if ctx.device == "gpu":
            import mlx.core as mx

            return mx.minimum(a, b)
        import numpy as np

        return np.minimum(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad * (a <= b)
        grad_b = grad * (a > b)
        return (grad_a, grad_b)


class Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if ctx.device == "gpu":
            import mlx.core as mx

            return mx.maximum(a, b)
        import numpy as np

        return np.maximum(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad * (a >= b)
        grad_b = grad * (a < b)
        return (grad_a, grad_b)


class GetItem(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_for_backward(a.shape, key)
        return a[key]

    @staticmethod
    def backward(ctx, grad):
        orig_shape, key = ctx.saved_tensors
        zeros = backend.zeros(orig_shape, ctx.device)
        if ctx.device == "gpu":
            import mlx.core as mx
            import numpy as np

            z_np = np.zeros(orig_shape, dtype=np.float32)
            z_np[key] = np.array(grad)
            return (mx.array(z_np), None)
        else:
            zeros[key] = grad
            return (zeros, None)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        y = backend.sigmoid(a, ctx.device)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return (grad * y * (1.0 - y),)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        y = backend.lib(ctx.device).tanh(a)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return (grad * (1.0 - y**2),)


class GELU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        import numpy as np

        if ctx.device == "gpu":
            import mlx.core as mx

            return (
                0.5
                * a
                * (
                    1.0
                    + mx.tanh(
                        mx.array(np.sqrt(2.0 / np.pi)) * (a + 0.044715 * mx.power(a, 3))
                    )
                )
            )
        return (
            0.5
            * a
            * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * np.power(a, 3))))
        )

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        import numpy as np

        if ctx.device == "gpu":
            import mlx.core as mx

            c1 = mx.array(np.sqrt(2.0 / np.pi))
            c2 = mx.array(0.044715)
            x_cube = mx.power(a, 3)
            inner = c1 * (a + c2 * x_cube)
            tanh_inner = mx.tanh(inner)
            dtanh = 1.0 - mx.power(tanh_inner, 2)
            dinner = c1 * (1.0 + 3.0 * c2 * mx.power(a, 2))
            return (grad * (0.5 * (1.0 + tanh_inner) + 0.5 * a * dtanh * dinner),)
        else:
            c1 = np.sqrt(2.0 / np.pi)
            c2 = 0.044715
            x_cube = np.power(a, 3)
            inner = c1 * (a + c2 * x_cube)
            tanh_inner = np.tanh(inner)
            dtanh = 1.0 - np.power(tanh_inner, 2)
            dinner = c1 * (1.0 + 3.0 * c2 * np.power(a, 2))
            return (grad * (0.5 * (1.0 + tanh_inner) + 0.5 * a * dtanh * dinner),)


class Softmax(Function):
    @staticmethod
    def forward(ctx, a):
        y = backend.softmax(a, axis=-1, device=ctx.device)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        sum_gy = backend.sum(grad * y, axis=-1, keepdims=True, device=ctx.device)
        return (y * (grad - sum_gy),)


class Trace(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.shape)
        if ctx.device == "gpu":
            import mlx.core as mx

            return mx.trace(a)
        import numpy as np

        return np.trace(a)

    @staticmethod
    def backward(ctx, grad):
        (shape,) = ctx.saved_tensors
        import numpy as np

        grad_out = np.zeros(shape, dtype=np.float32)
        min_dim = min(shape[-2], shape[-1])
        if ctx.device == "gpu":
            import mlx.core as mx

            grad_out = mx.array(grad_out)
            for i in range(min_dim):
                grad_out[..., i, i] = grad
        else:
            for i in range(min_dim):
                grad_out[..., i, i] = grad
        return (grad_out,)


class AsStrided(Function):
    @staticmethod
    def forward(ctx, a, shape, strides, storage_offset):
        ctx.save_for_backward(a.shape, shape, strides, storage_offset)
        itemsize = a.itemsize if hasattr(a, "itemsize") else 4
        byte_strides = tuple(s * itemsize for s in strides)

        if ctx.device == "gpu":
            import mlx.core as mx

            # Simple fallback if mlx as_strided needs special handling
            flat_array = mx.flatten(a)
            if storage_offset > 0:
                flat_array = flat_array[storage_offset:]
            return mx.as_strided(
                flat_array, shape=shape, strides=byte_strides, storage_offset=0
            )

        import numpy.lib.stride_tricks as np_stride_tricks

        flat_array = a.flatten()
        if storage_offset > 0:
            flat_array = flat_array[storage_offset:]
        return np_stride_tricks.as_strided(
            flat_array, shape=shape, strides=byte_strides
        )

    @staticmethod
    def backward(ctx, grad):
        orig_shape, shape, strides, storage_offset = ctx.saved_tensors
        import numpy as np

        # Exact backward for as_strided is complex, falling back to primitive zeroing + accumulation
        grad_a = np.zeros(orig_shape, dtype=np.float32)
        flat_grad_a = grad_a.reshape(-1)

        # This is a naive heuristic (since true general strided backward requires careful scatter_add)
        import itertools
        from energizer.backend import to_numpy

        np_grad = backend.to_numpy(grad, ctx.device)

        ranges = [range(s) for s in shape]
        for it in itertools.product(*ranges):
            flat_idx = storage_offset + sum(i * s for i, s in zip(it, strides))
            flat_grad_a[flat_idx] += np_grad[it]

        return (backend.array(grad_a, ctx.device), None, None, None)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unbroadcast_grad(grad, shape, device: str):
    """Sum out broadcasted dimensions so grad matches shape."""
    grad = backend.array(grad, device)
    g_shape = grad.shape if hasattr(grad, "shape") else ()
    if g_shape == shape:
        return grad

    if len(shape) == 0:
        return backend.sum(grad, device=device)

    ndim_diff = len(g_shape) - len(shape)
    if ndim_diff > 0:
        for _ in range(ndim_diff):
            grad = backend.sum(grad, axis=0, device=device)

    g_shape = grad.shape if hasattr(grad, "shape") else ()
    for i, (g_dim, s_dim) in enumerate(zip(g_shape, shape)):
        if s_dim == 1 and g_dim > 1:
            grad = backend.sum(grad, axis=i, keepdims=True, device=device)

    # Some backends may drop shape info entirely if shape is () but grad is scalar
    if hasattr(grad, "reshape"):
        grad = grad.reshape(shape)
    else:
        import numpy as np

        grad = np.array(grad).reshape(shape)

    return grad


def _is_mlx(data) -> bool:
    try:
        import mlx.core as mx

        return isinstance(data, mx.array)
    except ImportError:
        return False


def _wrap(value, reference: Tensor) -> Tensor:
    """Ensure scalars / plain arrays are wrapped as Tensors on the right device."""
    if isinstance(value, Tensor):
        return value
    return Tensor(
        backend.array(value, reference.device),
        device=reference.device,
        requires_grad=False,
    )
