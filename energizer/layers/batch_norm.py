from energizer.backend import normalize_device
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np
from energizer._mlx import mx


class _BatchNorm(Module):
    """Batch norm over channel dim 1, built from differentiable Tensor ops.

    Running statistics are plain arrays on the input's device and are updated
    outside the autograd graph (unbiased variance, like PyTorch).
    """

    _ndims = ()  # accepted input ranks

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(np.ones(num_features), requires_grad=True)
        self.beta = Parameter(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.training = True

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _on(arr, device):
        if device == "mlx":
            return arr if isinstance(arr, mx.array) else mx.array(np.asarray(arr))
        return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

    def _check(self, x: Tensor):
        if len(x.data.shape) not in self._ndims:
            raise ValueError(
                f"{type(self).__name__} expects {self._ndims}D input, "
                f"got {len(x.data.shape)}D"
            )

    def forward(self, x: Tensor) -> Tensor:
        self._check(x)
        nd = len(x.data.shape)
        axes = tuple(i for i in range(nd) if i != 1)
        view = [1] * nd
        view[1] = self.num_features
        view = tuple(view)

        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=axes, keepdims=True)
            n = x.data.size // self.num_features
            unbias = n / max(n - 1, 1)
            m = self.momentum
            rm = self._on(self.running_mean, x.device)
            rv = self._on(self.running_var, x.device)
            self.running_mean = (1 - m) * rm + m * mean.data.reshape(-1)
            self.running_var = (1 - m) * rv + m * unbias * var.data.reshape(-1)
            out = centered / ((var + self.eps) ** 0.5)
        else:
            rm = Tensor(
                self._on(self.running_mean, x.device).reshape(view), device=x.device
            )
            rv = Tensor(
                self._on(self.running_var, x.device).reshape(view), device=x.device
            )
            out = (x - rm) / ((rv + self.eps) ** 0.5)

        return out * self.gamma.reshape(view) + self.beta.reshape(view)

    def to(self, device: str):
        device = normalize_device(device)
        super().to(device)
        self.running_mean = self._on(self.running_mean, device)
        self.running_var = self._on(self.running_var, device)
        return self

    def eval(self):
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode


class BatchNorm1d(_BatchNorm):
    _ndims = (2, 3)


class BatchNorm2d(_BatchNorm):
    _ndims = (4,)
