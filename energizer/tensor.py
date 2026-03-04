import numpy as np
import mlx.core as mx
from typing import Union, Optional

import energizer.derivatives as dv
from .function import Function


class Tensor:
    """
    A tensor is a multi-dimensional array.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        grad_fn: Optional["Function"] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

        if self.device == "cpu":
            self._init_cpu(data)
        elif self.device == "gpu":
            self._init_mlx(data)

    def __str__(self):
        # Simple debugger-style representation showing device and backend
        backend = "mlx" if isinstance(self.data, mx.array) else "numpy"
        data_str = np.array2string(
            np.array(self.data),
            separator=", ",
            precision=4,
            suppress_small=False,
            floatmode="maxprec_equal",
        )
        return (
            f"tensor({data_str}, "
            f"device={self.device}, "
            f"backend={backend}, "
            f"requires_grad={self.requires_grad})"
        )

    def _init_cpu(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, mx.array):
            self.data = np.array(data)
        elif isinstance(data, Tensor):
            if data.device == "gpu":
                self.data = np.array(data.data)
            else:
                self.data = data.data.copy()
        else:
            self.data = np.array(data)

        self.shape = self.data.shape

    def _init_mlx(self, data):
        if isinstance(data, mx.array):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = mx.array(data)
        elif isinstance(data, Tensor):
            if data.device == "gpu":
                self.data = data.data
            else:
                self.data = mx.array(data.data)
        else:
            self.data = mx.array(data)

        self.shape = self.data.shape
        self.device = "gpu"

    def cpu(self) -> "Tensor":
        if self.device == "cpu":
            return self
        return Tensor(
            np.array(self.data), requires_grad=self.requires_grad, device="cpu"
        )

    def gpu(self) -> "Tensor":
        if self.device == "gpu":
            return self
        return Tensor(
            mx.array(self.data), requires_grad=self.requires_grad, device="gpu"
        )

    def to(self, device: str) -> "Tensor":
        if self.device == device:
            return self
        if device == "cpu":
            return self.cpu()
        elif device == "gpu":
            return self.gpu()
        else:
            raise ValueError(f"Invalid device: {device}")

    def numpy(self) -> np.ndarray:
        if isinstance(self.data, mx.array):
            return np.array(self.data)
        return self.data.copy()

    def mlx(self) -> mx.array:
        if isinstance(self.data, mx.array):
            return self.data
        return mx.array(self.data)

    def _get_other_data(self, other: Union[int, float, "Tensor"]):
        if isinstance(other, (int, float)):
            return other
        elif isinstance(other, Tensor):
            if self.device == "cpu":
                return other.cpu().data
            elif self.device == "gpu":
                return other.gpu().data
        # Handle MLX arrays and numpy arrays directly
        elif isinstance(other, mx.array):
            if self.device == "cpu":
                return np.array(other)
            else:
                return other
        elif isinstance(other, np.ndarray):
            if self.device == "gpu":
                return mx.array(other)
            else:
                return other
        # Fallback: try to access .data if it exists
        elif hasattr(other, "data"):
            return other.data
        else:
            return other

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self

    def __mul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        return Tensor(
            self.data * other_data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.mul_backward, [self, other]),
            device=self.device,
        )

    def __rmul__(self, other: Union[int, float]) -> "Tensor":
        return Tensor(
            other * self.data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.mul_backward, [self, other]),
            device=self.device,
        )

    def __add__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        return Tensor(
            self.data + other_data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.add_backward, [self, other]),
            device=self.device,
        )

    def __radd__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        return Tensor(
            self.data - other_data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.sub_backward, [self, other]),
            device=self.device,
        )

    def __neg__(self) -> "Tensor":
        return Tensor(
            (
                isinstance(self.data, (int, float)) - self.data
                if isinstance(self.data, (int, float))
                else -self.data
            ),
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.neg_backward, [self]),
            device=self.device,
        )

    def __truediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        return Tensor(
            self.data / other_data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.truediv_backward, [self, other]),
            device=self.device,
        )

    def __rtruediv__(self, other: Union[int, float]) -> "Tensor":
        other = self._get_other_data(other)
        return Tensor(
            other / self.data,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.truediv_backward, [other, self]),
            device=self.device,
        )

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other_data = self._get_other_data(other)
        return Tensor(
            self.data @ other_data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=Function(dv.matmul_backward, [self, other]),
            device=self.device,
        )

    def __pow__(self, other: Union[int, float]) -> "Tensor":
        return Tensor(
            self.data**other,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.pow_backward, [self, other]),
            device=self.device,
        )

    def __rmatmul__(self, other: "Tensor") -> "Tensor":
        return self.__matmul__(other)

    def __getitem__(self, key):
        return Tensor(
            self.data[key],
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.getitem_backward, [self, key]),
            device=self.device,
        )

    def __setitem__(self, key, value: Union[int, float, "Tensor"]):
        if isinstance(value, Tensor):
            value_data = value.data
        else:
            value_data = value

        if isinstance(self.data, mx.array):
            temp_np = np.array(self.data)
            temp_np[key] = value_data
            new_data = mx.array(temp_np)
        else:
            new_data = self.data.copy()
            new_data[key] = value_data

        if self.requires_grad:
            self.grad_fn = Function(dv.setitem_backward, [self, key, value])

        self.data = new_data

    @property
    def T(self) -> "Tensor":
        return Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.transpose_backward, [self]),
            device=self.device,
        )

    @property
    def size(self):
        return self.data.shape

    def item(self) -> Union[int, float]:
        if self.data.size != 1:
            raise ValueError("Tensor must be a single element")

        scalar_value = self.data.item()
        if self.requires_grad:
            self.grad_fn = Tensor(
                scalar_value,
                requires_grad=self.requires_grad,
                grad_fn=Function(dv.item_backward, [self]),
                device=self.device,
            )
        return scalar_value

    def copy(self) -> "Tensor":
        return Tensor(
            self.data.copy(), requires_grad=self.requires_grad, grad_fn=self.grad_fn
        )

    def sum(self, dim: Optional[int] = None, axis: Optional[int] = None) -> "Tensor":
        if axis is not None:
            dim = axis
        if isinstance(self.data, mx.array):
            result = mx.sum(self.data, axis=dim)
        else:
            result = np.sum(self.data, axis=dim)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.sum_backward, [self, dim]),
            device=self.device,
        )

    def mean(self, dim: Optional[int] = None, axis: Optional[int] = None) -> "Tensor":
        if axis is not None:
            dim = axis
        if isinstance(self.data, mx.array):
            result = mx.mean(self.data, axis=dim)
        else:
            result = np.mean(self.data, axis=dim)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.mean_backward, [self, dim]),
            device=self.device,
        )

    def squeeze(
        self, dim: Optional[int] = None, axis: Optional[int] = None
    ) -> "Tensor":
        if axis is not None:
            dim = axis
        if isinstance(self.data, mx.array):
            result = self.data.squeeze(axis=dim)
        else:
            result = self.data.squeeze(axis=dim)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.squeeze_backward, [self, dim]),
            device=self.device,
        )

    def exp(self) -> "Tensor":
        if isinstance(self.data, mx.array):
            result = self.data.exp()
        else:
            result = np.exp(self.data)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.exp_backward, [self]),
            device=self.device,
        )

    def log(self) -> "Tensor":
        if isinstance(self.data, mx.array):
            result = self.data.log()
        else:
            result = np.log(self.data)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.log_backward, [self]),
            device=self.device,
        )

    def clamp(self, min: Union[int, float], max: Union[int, float]) -> "Tensor":
        if isinstance(self.data, mx.array):
            result = mx.clip(self.data, min, max)
        else:
            result = np.clip(self.data, min, max)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.clamp_backward, [self, min, max]),
            device=self.device,
        )

    def minimum(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        if isinstance(self.data, mx.array):
            result = mx.minimum(self.data, other_data)
        else:
            result = np.minimum(self.data, other_data)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.minimum_backward, [self, other]),
            device=self.device,
        )

    def maximum(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        other_data = self._get_other_data(other)
        if isinstance(self.data, mx.array):
            result = mx.maximum(self.data, other_data)
        else:
            result = np.maximum(self.data, other_data)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.maximum_backward, [self, other]),
            device=self.device,
        )

    def reshape(self, shape: tuple) -> "Tensor":
        if isinstance(self.data, mx.array):
            result = self.data.reshape(shape)
        else:
            result = self.data.reshape(shape)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.reshape_backward, [self, shape]),
            device=self.device,
        )

    def view(self, shape: tuple) -> "Tensor":
        return self.reshape(shape)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        if self.device == "gpu":
            result = mx.swapaxes(self.data, dim0, dim1)
        else:
            result = np.swapaxes(self.data, dim0, dim1)
        return Tensor(
            result,
            requires_grad=self.requires_grad,
            grad_fn=Function(dv.transpose_backward, [self, dim0, dim1]),
            device=self.device,
        )

    @staticmethod
    def randn(*args, device: str = "cpu", **kwargs):
        if device == "gpu":
            data = mx.random.normal(shape=args, **kwargs)
        else:
            data = np.random.randn(*args, **kwargs)

        return Tensor(data, requires_grad=False, device=device)

    @staticmethod
    def zeros(shape, device: str = "cpu"):
        if device == "gpu":
            data = mx.zeros(shape)
        else:
            data = np.zeros(shape)
        return Tensor(data, requires_grad=False, device=device)

    @staticmethod
    def ones(shape, device: str = "cpu"):
        if device == "gpu":
            data = mx.ones(shape)
        else:
            data = np.ones(shape)
        return Tensor(data, requires_grad=False, device=device)

    def backward(self, grad_outputs=None):
        if grad_outputs is None:
            if isinstance(self.data, mx.array):
                grad_outputs = Tensor(mx.ones_like(self.data), device=self.device)
            else:
                grad_outputs = Tensor(np.ones_like(self.data), device=self.device)
        else:
            # Convert numpy/mx array to Tensor if needed
            if not isinstance(grad_outputs, Tensor):
                grad_outputs = Tensor(grad_outputs, device=self.device)
            # Ensure grad_outputs is on the same device as self
            elif self.device != grad_outputs.device:
                grad_outputs = grad_outputs.to(self.device)

        if self.requires_grad and self.grad_fn is not None:
            self.grad_fn.backward([grad_outputs])


def tensor(data, requires_grad=False, device="cpu"):
    return Tensor(data, requires_grad=requires_grad, device=device)
