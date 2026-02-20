from energizer.neural_network import Module
from energizer.tensor import Tensor
from energizer.function import Function
import energizer.derivatives as dv
import numpy as np


class Reshape(Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")

        new_shape = list(self.shape)

        if -1 in new_shape:
            total_elements = np.prod(x.shape)
            known_elements = np.prod([abs(d) for d in new_shape if d != -1])
            inferred_dim = total_elements // known_elements
            new_shape[new_shape.index(-1)] = inferred_dim

        new_shape = tuple(new_shape)

        if x.device == "gpu":
            import mlx.core as mx

            if isinstance(x.data, mx.array):
                reshaped_data = mx.reshape(x.data, new_shape)
            else:
                reshaped_data = mx.reshape(mx.array(x.data), new_shape)
        else:
            if isinstance(x.data, np.ndarray):
                reshaped_data = x.data.reshape(new_shape)
            else:
                reshaped_data = np.array(x.data).reshape(new_shape)

        return Tensor(
            reshaped_data,
            requires_grad=x.requires_grad,
            grad_fn=Function(dv.reshape_backward, [x]) if x.requires_grad else None,
            device=x.device,
        )
