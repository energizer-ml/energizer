from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class Trim(Module):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: Tensor) -> Tensor:
        h_start = self.start
        h_end = -self.end if self.end != 0 else None
        w_start = self.start
        w_end = -self.end if self.end != 0 else None
        trimmed = x.data[:, :, h_start:h_end, w_start:w_end]
        return Tensor(trimmed, requires_grad=x.requires_grad)
