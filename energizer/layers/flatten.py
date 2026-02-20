from energizer.neural_network import Module
from energizer.tensor import Tensor
from energizer.function import Function
import energizer.derivatives as dv
import numpy as np


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

        flattened = x.data.reshape(batch_size, -1)
        return Tensor(
            flattened,
            requires_grad=x.requires_grad,
            grad_fn=Function(dv.reshape_backward, [x]) if x.requires_grad else None,
            device=x.device,
        )
