from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np
from energizer._mlx import mx


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.zeros(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if len(x.data.shape) == 2:
                mean = x.data.mean(axis=0)
                var = x.data.var(axis=0)
            else:
                mean = x.data.mean(axis=(0, 2))
                var = x.data.var(axis=(0, 2))

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

            x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

        else:
            x_normalized = (x.data - self.running_mean) / np.sqrt(
                self.running_var + self.eps
            )

        if len(x.data.shape) == 2:
            result = x_normalized * self.gamma + self.beta
        else:
            gamma_reshaped = self.gamma.data.reshape(1, -1, 1)
            beta_reshaped = self.beta.data.reshape(1, -1, 1)
            result = x_normalized * gamma_reshaped + beta_reshaped

        return Tensor(result, requires_grad=x.requires_grad)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter(np.ones(num_features), requires_grad=True)
        self.beta = Parameter(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if len(x.data.shape) != 4:
            raise ValueError("BatchNorm2d only supports 4D input tensors")

        batch_size, channels, height, width = x.data.shape

        if self.training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

            # reshape for broadcasting
            mean_reshaped = mean.reshape(1, channels, 1, 1)
            var_reshaped = var.reshape(1, channels, 1, 1)

        else:
            mean_reshaped = self.running_mean.reshape(1, channels, 1, 1)
            var_reshaped = self.running_var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / (var_reshaped + self.eps) ** 0.5

        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        result = x_normalized * gamma_reshaped + beta_reshaped

        return Tensor(result, requires_grad=x.requires_grad, device=x.device)

    def to(self, device: str):
        super().to(device)
        if device == "gpu" and mx is not None:
            if isinstance(self.running_mean, np.ndarray):
                self.running_mean = mx.array(self.running_mean)
            if isinstance(self.running_var, np.ndarray):
                self.running_var = mx.array(self.running_var)
        else:
            if not isinstance(self.running_mean, np.ndarray):
                self.running_mean = np.array(self.running_mean)
            if not isinstance(self.running_var, np.ndarray):
                self.running_var = np.array(self.running_var)
        return self

    def eval(self):
        self.training = False

    def train(self):
        self.training = True
