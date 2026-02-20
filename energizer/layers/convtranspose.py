from energizer.neural_network import Module, Parameter
import energizer
import numpy as np
import mlx.core as mx
from energizer.tensor import Tensor


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

        if device == "gpu":
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

        batch_size, in_channels, height, width = x.shape

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding
        dh, dw = self.dilation

        if (dh, dw) != (1, 1):
            raise NotImplementedError(
                "ConvTranspose2d dilation != 1 is not supported yet"
            )

        if self.device == "gpu":
            if not isinstance(x.data, mx.array):
                x_data = mx.array(x.data)
            else:
                x_data = x.data

            if not isinstance(self.weight.data, mx.array):
                weight_data = mx.array(self.weight.data)
            else:
                weight_data = self.weight.data

            x_channels_last = mx.transpose(x_data, (0, 2, 3, 1))

            weight_mlx = mx.transpose(weight_data, (1, 2, 3, 0))

            kwargs = {
                "stride": (sh, sw),
                "padding": (ph, pw),
                "dilation": (dh, dw),
                "groups": self.groups,
            }
            if (oph, opw) != (0, 0):
                kwargs["output_padding"] = (oph, opw)

            output = mx.conv_transpose2d(x_channels_last, weight_mlx, **kwargs)
            output = mx.transpose(output, (0, 3, 1, 2))
        else:
            x_data = x.data if isinstance(x.data, np.ndarray) else np.array(x.data)
            weight_data = (
                self.weight.data
                if isinstance(self.weight.data, np.ndarray)
                else np.array(self.weight.data)
            )

            out_height = (height - 1) * sh - 2 * ph + (kh - 1) + oph + 1
            out_width = (width - 1) * sw - 2 * pw + (kw - 1) + opw + 1

            up_h = (height - 1) * sh + 1
            up_w = (width - 1) * sw + 1
            upsampled = np.zeros(
                (batch_size, in_channels, up_h, up_w), dtype=x_data.dtype
            )
            upsampled[:, :, ::sh, ::sw] = x_data

            pad_h = kh - 1 - ph
            pad_w = kw - 1 - pw
            padded = np.pad(
                upsampled,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

            flipped_weight = np.flip(weight_data, axis=(2, 3))
            conv_h = padded.shape[2] - kh + 1
            conv_w = padded.shape[3] - kw + 1

            output = np.zeros(
                (batch_size, self.out_channels, conv_h, conv_w), dtype=x_data.dtype
            )

            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for ic in range(in_channels):
                        kernel = flipped_weight[ic, oc]
                        for i in range(conv_h):
                            for j in range(conv_w):
                                patch = padded[b, ic, i : i + kh, j : j + kw]
                                output[b, oc, i, j] += np.sum(patch * kernel)

            if oph or opw:
                output = np.pad(
                    output, ((0, 0), (0, 0), (0, oph), (0, opw)), mode="constant"
                )

            output = output[:, :, :out_height, :out_width]

        if self.bias is not None:
            if self.device == "gpu":
                if isinstance(output, mx.array):
                    bias_data = (
                        self.bias.mlx()
                        if hasattr(self.bias, "mlx")
                        else mx.array(self.bias.data)
                    )
                    output += bias_data.reshape(1, -1, 1, 1)
                else:
                    output += np.array(self.bias.data).reshape(1, -1, 1, 1)
            else:
                bias_data = (
                    self.bias.data
                    if isinstance(self.bias.data, np.ndarray)
                    else np.array(self.bias.data)
                )
                output += bias_data.reshape(1, -1, 1, 1)

        return Tensor(output, requires_grad=x.requires_grad, device=self.device)

    def to(self, device: str):
        self.device = device
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self
