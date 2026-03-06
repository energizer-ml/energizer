from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer.autograd as autograd


class Sigmoid(Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return autograd.Sigmoid.apply(x)
