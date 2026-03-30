from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np
from energizer._mlx import mx


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.gamma = Tensor(
            np.ones(normalized_shape), requires_grad=True, device=device
        )
        self.beta = Tensor(
            np.zeros(normalized_shape), requires_grad=True, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.device == "gpu":
            mean = mx.mean(x.data, axis=-1, keepdims=True)
            var = mx.var(x.data, axis=-1, keepdims=True)
        else:
            mean = x.data.mean(axis=-1, keepdims=True)
            var = x.data.var(axis=-1, keepdims=True)

        x_normalized = (x.data - mean) / (var + self.eps) ** 0.5

        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma.data + self.beta.data

        return Tensor(x_normalized, requires_grad=x.requires_grad, device=x.device)
