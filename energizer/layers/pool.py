from energizer.autograd import Function
from energizer.backend import normalize_device
from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np
from energizer._mlx import mx

# Pooling is a loop over the kh*kw window offsets; each offset is one strided
# slice of the whole (B, C, H, W) batch, so the same code runs on NumPy and MLX
# and the backward pass scatters gradients back with strided slice-adds.


def _xp(ctx):
    return mx if ctx.device == "mlx" else np


def _zeros(ctx, shape):
    return mx.zeros(shape) if ctx.device == "mlx" else np.zeros(shape, np.float32)


def _pad(ctx, x, padding, value):
    ph, pw = padding
    if ph == 0 and pw == 0:
        return x
    width = ((0, 0), (0, 0), (ph, ph), (pw, pw))
    if ctx.device == "mlx":
        return mx.pad(x, width, constant_values=value)
    return np.pad(x, width, constant_values=value)


def _offsets(kernel):
    return [(i, j) for i in range(kernel[0]) for j in range(kernel[1])]


def _window(x, i, j, stride, out_hw):
    sh, sw = stride
    oh, ow = out_hw
    return x[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw]


def _out_hw(h, w, kernel, stride):
    return (h - kernel[0]) // stride[0] + 1, (w - kernel[1]) // stride[1] + 1


class MaxPool2dFn(Function):
    @staticmethod
    def forward(ctx, x, kernel, stride, padding):
        xp = _xp(ctx)
        x = _pad(ctx, x, padding, float(np.finfo(np.float32).min))
        out_hw = _out_hw(x.shape[2], x.shape[3], kernel, stride)
        best, arg = None, None
        for k, (i, j) in enumerate(_offsets(kernel)):
            win = _window(x, i, j, stride, out_hw)
            if best is None:
                best, arg = win, xp.zeros(win.shape, dtype=xp.int32)
            else:
                better = win > best  # strict: first maximum wins, like PyTorch
                best = xp.where(better, win, best)
                arg = xp.where(better, k, arg)
        ctx.save_for_backward(arg, x.shape, kernel, stride, padding, out_hw)
        return best

    @staticmethod
    def backward(ctx, g):
        arg, shape, kernel, stride, padding, out_hw = ctx.saved_tensors
        gx = _zeros(ctx, shape)
        sh, sw = stride
        oh, ow = out_hw
        for k, (i, j) in enumerate(_offsets(kernel)):
            contrib = g * (arg == k)
            gx[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += contrib
        ph, pw = padding
        if ph or pw:
            gx = gx[:, :, ph : shape[2] - ph, pw : shape[3] - pw]
        return gx, None, None, None


class AvgPool2dFn(Function):
    @staticmethod
    def forward(ctx, x, kernel, stride, padding):
        x = _pad(ctx, x, padding, 0.0)
        out_hw = _out_hw(x.shape[2], x.shape[3], kernel, stride)
        total = None
        for i, j in _offsets(kernel):
            win = _window(x, i, j, stride, out_hw)
            total = win if total is None else total + win
        ctx.save_for_backward(x.shape, kernel, stride, padding, out_hw)
        return total / float(kernel[0] * kernel[1])

    @staticmethod
    def backward(ctx, g):
        shape, kernel, stride, padding, out_hw = ctx.saved_tensors
        gx = _zeros(ctx, shape)
        sh, sw = stride
        oh, ow = out_hw
        share = g / float(kernel[0] * kernel[1])
        for i, j in _offsets(kernel):
            gx[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += share
        ph, pw = padding
        if ph or pw:
            gx = gx[:, :, ph : shape[2] - ph, pw : shape[3] - pw]
        return gx, None, None, None


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.data.shape) != 4:
            raise ValueError("MaxPool2d only supports 4D input tensors")
        return MaxPool2dFn.apply(x, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return []


class AvgPool2d(Module):
    def __init__(
        self, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0)
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.data.shape) != 4:
            raise ValueError("AvgPool2d only supports 4D input tensors")
        return AvgPool2dFn.apply(x, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return []
