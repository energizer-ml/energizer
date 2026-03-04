from typing import Union
from energizer.neural_network import Module
import numpy as np

from energizer.tensor import Tensor
from energizer.functionnal import max


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return max(x)

    def extra_repr(self):
        return ""


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return max(x) + self.negative_slope * (x - max(x))

    def extra_repr(self):
        return ""
