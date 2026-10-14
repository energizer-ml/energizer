from energizer.neural_network import Module
from energizer.tensor import Tensor
from energizer.function import Function
import energizer.derivatives as dv
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        scale = 1.0 / (1.0 - self.p)
        if x.device == "mlx":
            import mlx.core as mx

            mask = (mx.random.uniform(shape=x.data.shape) > self.p).astype(
                mx.float32
            ) * scale
        else:
            mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) * scale
        # multiplying by a constant Tensor keeps the op differentiable
        return x * Tensor(mask, device=x.device)

    def eval(self):
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode
