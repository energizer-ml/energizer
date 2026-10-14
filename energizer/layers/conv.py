from energizer.backend import normalize_device
from energizer.autograd import Function
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np
import energizer.functionnal as F
from energizer._mlx import mx

# ── Differentiable convolution ───────────────────────────────────────────────
# One implementation (2-D); 1-D reuses it with a height-1 image.
#   cpu : NumPy sliding-window einsum, explicit col2im backward
#   mlx : mx.conv2d, backward through mx.vjp (stays on the Metal device)


def _conv2d_np(x, w, stride, padding):
    from numpy.lib.stride_tricks import sliding_window_view

    sh, sw = stride
    ph, pw = padding
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kh, kw = w.shape[2:]
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    out = np.einsum("bcijkl,ockl->boij", win, w, optimize=True)
    return out.astype(np.float32), win, xp.shape


def _conv2d_mx(x, w, stride, padding):
    # MLX convolutions are channels-last: NCHW -> NHWC, OIHW -> OHWI.
    y = mx.conv2d(
        x.transpose(0, 2, 3, 1),
        w.transpose(0, 2, 3, 1),
        stride=tuple(stride),
        padding=tuple(padding),
    )
    return y.transpose(0, 3, 1, 2)


def _col2im(gwin, xp_shape, x_shape, stride, padding):
    """Scatter window gradients (b,c,oh,ow,kh,kw) back onto the image."""
    sh, sw = stride
    ph, pw = padding
    oh, ow, kh, kw = gwin.shape[2:]
    gxp = np.zeros(xp_shape, dtype=np.float32)
    for k in range(kh):
        for l in range(kw):
            gxp[:, :, k : k + sh * oh : sh, l : l + sw * ow : sw] += gwin[
                :, :, :, :, k, l
            ]
    h, wd = x_shape[2:]
    return gxp[:, :, ph : ph + h, pw : pw + wd]


class _ConvFn(Function):
    """Shared forward/backward on NCHW tensors; subclasses adapt 1-D shapes."""

    @staticmethod
    def _run_forward(ctx, x, w, b, stride, padding):
        ctx.conv = (stride, padding)
        if ctx.device == "mlx":
            ctx.save_for_backward(x, w)
            y = _conv2d_mx(x, w, stride, padding)
        else:
            x = np.asarray(x, dtype=np.float32)
            w = np.asarray(w, dtype=np.float32)
            y, win, xp_shape = _conv2d_np(x, w, stride, padding)
            ctx.save_for_backward(win, w)
            ctx.conv = (stride, padding, xp_shape, x.shape)
        if b is not None:
            y = y + b.reshape(1, -1, 1, 1)
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def _run_backward(ctx, g):
        a, w = ctx.saved_tensors
        if ctx.device == "mlx":
            stride, padding = ctx.conv
            _, (gx, gw) = mx.vjp(
                lambda x_, w_: _conv2d_mx(x_, w_, stride, padding), [a, w], [g]
            )
            gb = g.sum(axis=(0, 2, 3)) if ctx.has_bias else None
            return gx, gw, gb

        stride, padding, xp_shape, x_shape = ctx.conv
        win = a
        g = np.asarray(g, dtype=np.float32)
        gw = np.einsum("bcijkl,boij->ockl", win, g, optimize=True)
        gwin = np.einsum("boij,ockl->bcijkl", g, w, optimize=True)
        gx = _col2im(gwin, xp_shape, x_shape, stride, padding)
        gb = g.sum(axis=(0, 2, 3)) if ctx.has_bias else None
        return gx, gw, gb


def _convT_mx(x, w, stride, padding, output_padding):
    # w: (Cin, Cout, kh, kw) -> MLX (Cout, kh, kw, Cin)
    y = mx.conv_transpose2d(
        x.transpose(0, 2, 3, 1),
        w.transpose(1, 2, 3, 0),
        stride=tuple(stride),
        padding=tuple(padding),
        output_padding=tuple(output_padding),
    )
    return y.transpose(0, 3, 1, 2)


class ConvTranspose2dFn(Function):
    """Transposed convolution = input-gradient of conv2d with weight (Cin, Cout, kh, kw)."""

    @staticmethod
    def forward(ctx, x, w, b, stride, padding, output_padding):
        ctx.conv = (stride, padding, output_padding)
        ctx.has_bias = b is not None
        if ctx.device == "mlx":
            ctx.save_for_backward(x, w)
            y = _convT_mx(x, w, stride, padding, output_padding)
        else:
            x = np.asarray(x, dtype=np.float32)
            w = np.asarray(w, dtype=np.float32)
            sh, sw = stride
            ph, pw = padding
            kh, kw = w.shape[2:]
            h, wd = x.shape[2:]
            ho = (h - 1) * sh - 2 * ph + kh + output_padding[0]
            wo = (wd - 1) * sw - 2 * pw + kw + output_padding[1]
            # the conv this inverts sees an (ho, wo) image padded by (ph, pw)
            xp_shape = (x.shape[0], w.shape[1], ho + 2 * ph, wo + 2 * pw)
            gwin = np.einsum("boij,ockl->bcijkl", x, w, optimize=True)
            y = _col2im(
                gwin, xp_shape, (x.shape[0], w.shape[1], ho, wo), stride, padding
            )
            ctx.save_for_backward(x, w)
        if b is not None:
            y = y + b.reshape(1, -1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        stride, padding, output_padding = ctx.conv
        if ctx.device == "mlx":
            _, (gx, gw) = mx.vjp(
                lambda x_, w_: _convT_mx(x_, w_, stride, padding, output_padding),
                [x, w],
                [g],
            )
        else:
            g = np.asarray(g, dtype=np.float32)
            # dx is the forward conv of g with the same weight; dw pairs windows of g with x
            gx, win, _ = _conv2d_np(g, w, stride, padding)
            gx = gx[:, :, : x.shape[2], : x.shape[3]]
            gw = np.einsum("bcijkl,boij->ockl", win, x, optimize=True)
        gb = g.sum(axis=(0, 2, 3)) if ctx.has_bias else None
        return gx, gw, gb, None, None, None


class Conv2dFn(_ConvFn):
    @staticmethod
    def forward(ctx, x, w, b, stride, padding):
        return _ConvFn._run_forward(ctx, x, w, b, stride, padding)

    @staticmethod
    def backward(ctx, g):
        gx, gw, gb = _ConvFn._run_backward(ctx, g)
        return gx, gw, gb, None, None


class Conv1dFn(_ConvFn):
    @staticmethod
    def forward(ctx, x, w, b, stride, padding):
        y = _ConvFn._run_forward(
            ctx, x[:, :, None, :], w[:, :, None, :], b, (1, stride), (0, padding)
        )
        return y[:, :, 0, :]

    @staticmethod
    def backward(ctx, g):
        gx, gw, gb = _ConvFn._run_backward(ctx, g[:, :, None, :])
        return gx[:, :, 0, :], gw[:, :, 0, :], gb, None, None


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
        if self.device == "mlx":
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
            if self.device == "mlx":
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
        return Conv1dFn.apply(input, self.weight, self.bias, self.stride, self.padding)

    def to(self, device: str):
        device = normalize_device(device)
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

        if self.device == "mlx":
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
            if self.device == "mlx":
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
        return Conv2dFn.apply(input, self.weight, self.bias, self.stride, self.padding)
