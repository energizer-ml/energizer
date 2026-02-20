from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.data.shape) != 4:
            raise ValueError("MaxPool2d only supports 4D input tensors")

        batch_size, channels, height, width = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x_padded = self._pad2d(x, self.padding)
            padded_height = height + 2 * ph
            padded_width = width + 2 * pw
        else:
            x_padded = x
            padded_height = height
            padded_width = width

        out_height = ((padded_height - kh) // sh) + 1
        out_width = ((padded_width - kw) // sw) + 1

        result = np.zeros((batch_size, channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * sh
                        w_start = j * sw
                        h_end = h_start + kh
                        w_end = w_start + kw

                        window = x_padded.data[b, c, h_start:h_end, w_start:w_end]
                        result[b, c, i, j] = np.max(window)

        return Tensor(result, requires_grad=x.requires_grad)

    def _pad2d(self, x: Tensor, padding: tuple) -> Tensor:
        ph, pw = padding
        batch_size, channels, height, width = x.data.shape
        if ph == 0 and pw == 0:
            return x
        min_val = (
            np.finfo(x.data.dtype).min
            if np.issubdtype(x.data.dtype, np.floating)
            else np.iinfo(x.data.dtype).min
        )
        padded_data = np.full(
            (batch_size, channels, height + 2 * ph, width + 2 * pw),
            min_val,
            dtype=x.data.dtype,
        )
        padded_data[:, :, ph : ph + height, pw : pw + width] = x.data
        return Tensor(padded_data, requires_grad=x.requires_grad)

    def parameters(self):
        return []


class AvgPool2d(Module):
    def __init__(
        self, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0)
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.data.shape) != 4:
            raise ValueError("AvgPool2d only supports 4D input tensors")

        batch_size, channels, height, width = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x_padded = self._pad2d(x, self.padding)
            padded_height = height + 2 * ph
            padded_width = width + 2 * pw
        else:
            x_padded = x
            padded_height = height
            padded_width = width

        out_height = ((padded_height - kh) // sh) + 1
        out_width = ((padded_width - kw) // sw) + 1

        result = np.zeros((batch_size, channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * sh
                        w_start = j * sw
                        h_end = h_start + kh
                        w_end = w_start + kw

                        window = x_padded.data[b, c, h_start:h_end, w_start:w_end]
                        result[b, c, i, j] = np.mean(window)

        return Tensor(result, requires_grad=x.requires_grad)

    def _pad2d(self, x: Tensor, padding: tuple) -> Tensor:
        ph, pw = padding
        batch_size, channels, height, width = x.data.shape
        if ph == 0 and pw == 0:
            return x
        padded_data = np.zeros(
            (batch_size, channels, height + 2 * ph, width + 2 * pw), dtype=x.data.dtype
        )
        padded_data[:, :, ph : ph + height, pw : pw + width] = x.data
        return Tensor(padded_data, requires_grad=x.requires_grad)

    def parameters(self):
        return []
