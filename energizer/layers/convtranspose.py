from energizer.backend import normalize_device
from energizer.neural_network import Module, Parameter
import energizer
import numpy as np
from energizer.tensor import Tensor
from energizer._mlx import mx


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = 1,
        padding: tuple = 0,
        output_padding: tuple = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: tuple = 1,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
        )
        self.groups = groups
        self.bias = bias
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.device = device

        weight_shape = (
            in_channels,
            out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        if device == "mlx":
            weight_data = mx.random.normal(weight_shape) * 0.01
            bias_data = mx.zeros(out_channels) if bias else None
        else:
            weight_data = np.random.randn(*weight_shape) * 0.01
            bias_data = np.zeros(out_channels) if bias else None

        self.weight = Parameter(weight_data, requires_grad=True, device=device)

        if bias:
            self.bias = Parameter(bias_data, requires_grad=True, device=device)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        if self.dilation != (1, 1):
            raise NotImplementedError(
                "ConvTranspose2d dilation != 1 is not supported yet"
            )
        if self.groups != 1:
            raise NotImplementedError(
                "ConvTranspose2d groups != 1 is not supported yet"
            )
        from energizer.layers.conv import ConvTranspose2dFn

        return ConvTranspose2dFn.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding
        )

    def to(self, device: str):
        device = normalize_device(device)
        self.device = device
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self
