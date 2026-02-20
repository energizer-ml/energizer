from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class MSELoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = (input - target) ** 2
        if self.reduction == "sum":
            return diff.sum()
        return diff.mean()

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)


class CrossEntropyLoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return -target * np.log(input)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)
