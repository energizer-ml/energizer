from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer.autograd as autograd


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.data.shape[0]
        if self.start_dim < 0:
            self.start_dim = len(x.data.shape) + self.start_dim
        if self.end_dim < 0:
            self.end_dim = len(x.data.shape) + self.end_dim

        new_shape = (batch_size, -1)
        return autograd.Reshape.apply(x, new_shape)
