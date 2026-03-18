from energizer import tensor
from energizer.neural_network import Module, Parameter
import numpy as np
from energizer._mlx import mx


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device: str = "cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        if device == "gpu":
            self.weight = Parameter(
                mx.random.normal(shape=(out_features, in_features)) * 0.01,
                requires_grad=True,
                device=device,
            )
        else:
            self.weight = Parameter(
                np.random.randn(out_features, in_features) * 0.01,
                requires_grad=True,
                device=device,
            )
        if bias:
            if device == "gpu":
                self.bias = Parameter(
                    mx.zeros(out_features), requires_grad=True, device=device
                )
            else:
                self.bias = Parameter(
                    np.zeros(out_features), requires_grad=True, device=device
                )
        else:
            self.bias = None

    def forward(self, x):
        result = x @ self.weight.T
        if self.bias is not None:
            result = result + self.bias
        return result

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
