from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.inplace = inplace
        self.mask = None
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        scale = 1.0 / (1.0 - self.p)
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) * scale

        self.mask = mask

        if self.inplace:
            x.data *= mask
            return x
        return Tensor(x.data * mask, requires_grad=x.requires_grad, device=x.device)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True
