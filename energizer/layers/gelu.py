from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np
import mlx.core as mx


class GELU(Module):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def forward(self, x: Tensor) -> Tensor:
        if x.device == "gpu":
            return Tensor(
                0.5
                * x.data
                * (
                    1.0
                    + mx.tanh(
                        mx.sqrt(2 / mx.pi) * (x.data + 0.044715 * mx.power(x.data, 3))
                    )
                ),
                requires_grad=x.requires_grad,
                device=x.device,
            )
        else:
            return Tensor(
                0.5
                * x.data
                * (
                    1.0
                    + np.tanh(
                        np.sqrt(2 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3))
                    )
                ),
                requires_grad=x.requires_grad,
                device=x.device,
            )
