from ctypes import Union
from energizer.tensor import Tensor
import numpy as np
import numpy.lib.stride_tricks as np_stride_tricks
import mlx.core as mx

from energizer.function import Function
import energizer.derivatives as dv

def max(tensor : Tensor, floor: Union[int, float] = 0) -> Tensor:
    if tensor.device == 'gpu':
        return Tensor(mx.maximum(floor, tensor.data), requires_grad=tensor.requires_grad, grad_fn=Function(dv.max_backward, [tensor, floor]), device=tensor.device)
    return Tensor(np.maximum(floor, tensor.data), requires_grad=tensor.requires_grad, grad_fn=Function(dv.max_backward, [tensor, floor]), device=tensor.device)

def as_strided(tensor: Tensor, shape: tuple, strides: tuple, storage_offset: int = 0) -> 'Tensor':
    array = tensor.data

    itemsize = array.itemsize
    byte_strides = tuple(s * itemsize for s in strides)
    
    if storage_offset > 0:
        flat_array = array.flatten()
        if storage_offset >= len(flat_array):
            raise ValueError(f"storage_offset {storage_offset} is out of bounds for array of size {len(flat_array)}")
        offset_array = flat_array[storage_offset:]
        array = offset_array

    if tensor.device == 'gpu':
        strided_data = mx.as_strided(array, shape=shape, strides=byte_strides, storage_offset=storage_offset)
    else:
        strided_data = np_stride_tricks.as_strided(array, shape=shape, strides=byte_strides)

    return Tensor(strided_data, requires_grad=tensor.requires_grad, grad_fn=Function(dv.as_strided_backward, [tensor, shape, strides, storage_offset]), device=tensor.device)

def trace(tensor: Tensor) -> Tensor:
    if tensor.device == 'gpu':
        result = mx.trace(tensor.data)
    else:
        result = np.trace(tensor.data)
    return Tensor(result, requires_grad=tensor.requires_grad, grad_fn=Function(dv.trace_backward, [tensor]), device=tensor.device)
