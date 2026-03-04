from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x.data), requires_grad=x.requires_grad, device=x.device)
