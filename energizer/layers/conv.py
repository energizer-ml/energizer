from energizer.function import Function
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np
import mlx.core as mx
import energizer.functionnal as F
import energizer.derivatives as dv


class ConvNd(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if self.device == "gpu":
            std = mx.sqrt(2.0 / (in_channels * kernel_size))
            self.weight = Parameter(
                mx.random.normal(shape=(out_channels, in_channels, kernel_size)),
                requires_grad=True,
                device=self.device,
            )
        else:
            std = np.sqrt(2.0 / (in_channels * kernel_size))
            self.weight = Parameter(
                np.random.randn(out_channels, in_channels, kernel_size) * std,
                requires_grad=True,
                device=self.device,
            )
        if bias:
            if self.device == "gpu":
                self.bias = Parameter(
                    mx.zeros(out_channels), requires_grad=True, device=self.device
                )
            else:
                self.bias = Parameter(
                    np.zeros(out_channels), requires_grad=True, device=self.device
                )
        else:
            self.bias = None
        self._parameters["weight"] = self.weight
        if self.bias is not None:
            self._parameters["bias"] = self.bias

    def forward(self) -> Tensor:
        pass


class Conv1d(ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, input: Tensor) -> Tensor:
        self._input = input
        self._input_padded = (
            self._pad1d(input, self.padding) if self.padding > 0 else input
        )

        batch_size, in_channels, length = self._input_padded.data.shape
        out_length = ((length - self.kernel_size) // self.stride) + 1

        x_2d = self._input_padded.data.reshape(-1, length)

        windows = []
        for i in range(x_2d.shape[0]):
            row_windows = F.as_strided(
                Tensor(x_2d[i]),
                shape=(out_length, self.kernel_size),
                strides=(self.stride, 1),
            ).data
            windows.append(row_windows)

        windows_stacked = np.stack(windows, axis=0)

        windows_reshaped = (
            windows_stacked.reshape(
                batch_size, in_channels, out_length, self.kernel_size
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, out_length, -1)
        )

        weight_reshaped = self.weight.data.reshape(self.out_channels, -1)

        result = np.einsum("bij,oj->bio", windows_reshaped, weight_reshaped)
        result = result.transpose(0, 2, 1)

        if self.bias is not None:
            result += self.bias.data.reshape(1, -1, 1)

        return Tensor(
            result,
            requires_grad=input.requires_grad,
            grad_fn=Function(
                dv.conv1d_backward,
                [self._input, self.weight, self.bias],
                stride=self.stride,
                padding=self.padding,
            ),
        )

    def _pad1d(self, input: Tensor, padding: int) -> Tensor:
        batch_size, channels, length = input.data.shape
        padded_length = length + 2 * padding
        padded_data = np.zeros((batch_size, channels, padded_length))
        padded_data[:, :, padding : padding + length] = input.data
        return Tensor(padded_data, requires_grad=input.requires_grad)

    def _unfold1d(
        self, input: Tensor, kernel_size: int, stride: int, out_length: int
    ) -> Tensor:
        batch_size, channels, length = input.data.shape
        unfolded_data = np.zeros((batch_size, out_length, channels, kernel_size))
        for b in range(batch_size):
            for c in range(channels):
                channel_data = input.data[b, c, :]

                window = F.as_strided(
                    Tensor(channel_data),
                    shape=(out_length, kernel_size),
                    strides=(stride, 1),
                )

                unfolded_data[b, :, c, :] = window.data

        return Tensor(unfolded_data, requires_grad=input.requires_grad)

    def to(self, device: str):
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = True,
    ):
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

        if self.device == "gpu":
            std = mx.sqrt(
                2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
            )
            self.weight = Parameter(
                mx.random.normal(
                    shape=(
                        out_channels,
                        in_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )
                ),
                requires_grad=True,
                device=self.device,
            )
        else:
            std = np.sqrt(
                2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
            )
            self.weight = Parameter(
                np.random.randn(
                    out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
                )
                * std,
                requires_grad=True,
                device=self.device,
            )
        if bias:
            if self.device == "gpu":
                self.bias = Parameter(
                    mx.zeros(out_channels), requires_grad=True, device=self.device
                )
            else:
                self.bias = Parameter(
                    np.zeros(out_channels), requires_grad=True, device=self.device
                )
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if self.padding != (0, 0):
            input = self._pad2d(input, self.padding)

        batch_size, in_channels, height, width = input.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        out_height = ((height - kh) // sh) + 1
        out_width = ((width - kw) // sw) + 1

        windows = self._im2col(input, kh, kw, sh, sw, out_height, out_width)

        if input.device == "gpu":
            weight_reshaped = input.mlx().__class__(
                self.weight.data.reshape(self.out_channels, -1)
            )

            result = mx.zeros((batch_size, out_height * out_width, self.out_channels))
            for b in range(batch_size):
                result[b] = windows[b] @ weight_reshaped.T
            result = result.transpose(0, 2, 1).reshape(
                batch_size, self.out_channels, out_height, out_width
            )
            if self.bias is not None:
                result += self.bias.data.reshape(1, -1, 1, 1)
        else:
            weight_reshaped = self.weight.data.reshape(self.out_channels, -1)

            result = np.zeros((batch_size, out_height * out_width, self.out_channels))
            for b in range(batch_size):
                result[b] = windows[b] @ weight_reshaped.T

            result = result.transpose(0, 2, 1).reshape(
                batch_size, self.out_channels, out_height, out_width
            )

            if self.bias is not None:
                result += self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(result, requires_grad=input.requires_grad, device=input.device)

    def _pad2d(self, x: Tensor, padding: tuple) -> Tensor:
        ph, pw = padding
        batch_size, channels, height, width = x.data.shape
        if ph == 0 and pw == 0:
            return x
        padded_data = np.zeros((batch_size, channels, height + 2 * ph, width + 2 * pw))
        padded_data[:, :, ph : ph + height, pw : pw + width] = x.data
        return Tensor(padded_data, requires_grad=x.requires_grad, device=x.device)

    def _im2col(
        self,
        x: Tensor,
        kh: int,
        kw: int,
        sh: int,
        sw: int,
        out_height: int,
        out_width: int,
    ):
        batch_size, channels, height, width = x.data.shape
        if x.device == "gpu":
            unfolded_data = mx.zeros(
                (batch_size, out_height * out_width, channels * kh * kw)
            )
        else:
            unfolded_data = np.zeros(
                (batch_size, out_height * out_width, channels * kh * kw)
            )

        for b in range(batch_size):
            col_index = 0
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * sh
                    w_start = j * sw
                    window = x.data[
                        b, :, h_start : h_start + kh, w_start : w_start + kw
                    ]
                    if x.device == "gpu":
                        window_flat = window.reshape(-1)
                    else:
                        window_flat = window.ravel()
                    unfolded_data[b, col_index] = window_flat
                    col_index += 1

        return unfolded_data
