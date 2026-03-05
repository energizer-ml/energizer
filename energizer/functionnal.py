from typing import Union
from energizer.tensor import Tensor
import energizer.autograd as autograd
import numpy as np

def max(tensor: Tensor, floor: Union[int, float] = 0) -> Tensor:
    return tensor.maximum(floor)

def as_strided(
    tensor: Tensor, shape: tuple, strides: tuple, storage_offset: int = 0
) -> "Tensor":
    return autograd.AsStrided.apply(tensor, shape, strides, storage_offset)

def trace(tensor: Tensor) -> Tensor:
    return autograd.Trace.apply(tensor)

def tanh(tensor: Tensor) -> Tensor:
    return autograd.Tanh.apply(tensor)

def softmax(tensor: Tensor) -> Tensor:
    return autograd.Softmax.apply(tensor)
