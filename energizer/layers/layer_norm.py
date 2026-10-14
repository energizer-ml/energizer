from energizer.backend import normalize_device
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np


class LayerNorm(Module):
    """Layer normalisation over the last dimension, built from differentiable ops."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.gamma = Parameter(
            np.ones(normalized_shape), requires_grad=True, device=device
        )
        self.beta = Parameter(
            np.zeros(normalized_shape), requires_grad=True, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        centered = x - x.mean(axis=-1, keepdims=True)
        var = (centered * centered).mean(axis=-1, keepdims=True)
        out = centered / ((var + self.eps) ** 0.5)
        if self.elementwise_affine:
            out = out * self.gamma + self.beta
        return out
