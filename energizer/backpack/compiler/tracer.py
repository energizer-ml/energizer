from typing import Any


class IRNode:
    def __init__(self, op: str, inputs: list, output_shape: tuple, output_dtype: str):
        self.op = op
        self.inputs = inputs
        self.output_shape = output_shape
        self.output_dtype = output_dtype

    def __repr__(self):
        return f"IRNode(op='{self.op}', inputs={len(self.inputs)}, shape={self.output_shape}, dtype='{self.output_dtype}')"


class TraceData:
    def __init__(self, shape: tuple, dtype: str = "float32"):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape) if shape else ()
        self.dtype = dtype

    @property
    def size(self):
        import numpy as np
        return int(np.prod(self.shape)) if self.shape else 1


class Tracer:
    _active = None

    def __init__(self):
        self.nodes = []
        self._tensor_to_node: dict[int, IRNode] = {}  # id(Tensor) → IRNode that produced it

    def __enter__(self):
        Tracer._active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tracer._active = None

    @classmethod
    def is_tracing(cls):
        return cls._active is not None

    @classmethod
    def get(cls):
        return cls._active

    def trace(self, model, example_inputs):
        self.nodes = []
        self._tensor_to_node = {}
        with self:
            model(*example_inputs)
        return self.nodes

    def record(self, op_name: str, input_tensors: list, result) -> IRNode:
        """
        Record one op into the IR graph.
        Resolves each input: if it was produced by a previous op, store the IRNode
        reference. Otherwise store the raw Tensor (weight / constant).
        """
        resolved_inputs = []
        for t in input_tensors:
            if id(t) in self._tensor_to_node:
                resolved_inputs.append(self._tensor_to_node[id(t)])
            else:
                resolved_inputs.append(t)  # weight or constant

        node = IRNode(
            op=op_name,
            inputs=resolved_inputs,
            output_shape=result.shape,
            output_dtype="float32",
        )
        self.nodes.append(node)
        self._tensor_to_node[id(result)] = node
        return node

    @classmethod
    def infer_shape(cls, op: str, *args):
        import numpy as np

        shapes = []
        for x in args:
            if isinstance(x, TraceData):
                shapes.append(x.shape)
            elif hasattr(x, "shape"):
                shapes.append(x.shape)
            elif isinstance(x, (int, float, complex, bool)):
                shapes.append(())
            elif isinstance(x, (list, tuple)) and op not in (
                "Reshape", "AsStrided", "SumAxis", "Squeeze",
            ):
                shapes.append(np.array(x).shape)

        if op in ("Add", "Sub", "Mul", "Div", "Minimum", "Maximum", "Pow"):
            return np.broadcast_shapes(shapes[0], shapes[1])

        if op in ("Neg", "Exp", "Log", "Clamp", "Sigmoid", "Tanh", "GELU", "Softmax"):
            return shapes[0]

        if op == "MatMul":
            shape_a, shape_b = shapes[0], shapes[1]
            if len(shape_a) == 0 or len(shape_b) == 0:
                raise ValueError("MatMul requires at least 1D tensors")
            if len(shape_a) == 1 and len(shape_b) == 1:
                return ()
            elif len(shape_a) == 1:
                return shape_b[:-2] + (shape_b[-1],)
            elif len(shape_b) == 1:
                return shape_a[:-1]
            else:
                batch_shape = np.broadcast_shapes(shape_a[:-2], shape_b[:-2])
                return batch_shape + (shape_a[-2], shape_b[-1])

        if op in ("Sum", "Mean"):
            return ()

        if op == "SumAxis":
            axis = args[1]
            shape = list(shapes[0])
            for ax in sorted([axis] if isinstance(axis, int) else list(axis), reverse=True):
                shape.pop(ax)
            return tuple(shape)

        if op == "Squeeze":
            axis = args[1] if len(args) > 1 else None
            shape = list(shapes[0])
            if axis is None:
                return tuple(s for s in shape if s != 1)
            for ax in sorted([axis] if isinstance(axis, int) else list(axis), reverse=True):
                if shape[ax] == 1:
                    shape.pop(ax)
            return tuple(shape)

        if op == "Reshape":
            new_shape = list(args[1])
            if -1 in new_shape:
                idx = new_shape.index(-1)
                total = int(np.prod(shapes[0])) if shapes[0] else 1
                known = int(np.prod([s for s in new_shape if s != -1]))
                new_shape[idx] = total // known
            return tuple(new_shape)

        if op == "Transpose":
            return tuple(reversed(shapes[0]))

        if op == "GetItem":
            dummy = np.empty(shapes[0], dtype=np.float32)[args[1]]
            return dummy.shape

        if op == "Trace":
            return shapes[0][:-2] if len(shapes[0]) >= 2 else ()

        if op == "AsStrided":
            return args[1]

        return shapes[0] if shapes else ()