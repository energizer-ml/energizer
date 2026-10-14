from energizer.backend import normalize_device
from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer.autograd as autograd


class GELU(Module):
    def __init__(self, device: str = "cpu"):
        device = normalize_device(device)
        super().__init__(device=device)

    def forward(self, x: Tensor) -> Tensor:
        return autograd.GELU.apply(x)
