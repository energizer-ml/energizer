from energizer.neural_network import Module
from energizer.tensor import Tensor
from energizer.function import Function
import energizer.derivatives as dv
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


class GELU(Module):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def forward(self, x: Tensor) -> Tensor:
        if x.device == "gpu" and mx is not None:
            val = (
                0.5
                * x.data
                * (
                    1.0
                    + mx.tanh(
                        mx.array(np.sqrt(2.0 / np.pi))
                        * (x.data + 0.044715 * mx.power(x.data, 3))
                    )
                )
            )
        else:
            val = (
                0.5
                * x.data
                * (
                    1.0
                    + np.tanh(
                        np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3))
                    )
                )
            )

        return Tensor(
            val,
            requires_grad=x.requires_grad,
            grad_fn=Function(dv.gelu_backward, [x]),
            device=x.device,
        )
