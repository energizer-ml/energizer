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
        # Always generate mask in numpy; convert to the right backend below
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) * scale

        try:
            import mlx.core as mx
            if isinstance(x.data, mx.array):
                out = x.data * mx.array(mask)
            else:
                out = x.data * mask
        except ImportError:
            out = x.data * mask

        return Tensor(
            out,
            requires_grad=x.requires_grad,
            grad_fn=Function(dv.dropout_backward, [x, mask]),
            device=x.device,
        )

    def eval(self):
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode
