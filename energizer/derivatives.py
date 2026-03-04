from __future__ import annotations
from typing import Any
import energizer.tensor as ts
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


def _reduce_grad_to_shape(grad: Any, shape: tuple) -> Any:
    if shape == ():
        if mx and isinstance(grad, mx.array):
            return mx.sum(grad)
        return np.sum(grad)

    is_mlx = mx and isinstance(grad, mx.array)

    g = grad
    while len(getattr(g, "shape", ())) > len(shape):
        if is_mlx:
            g = mx.sum(g, axis=0)
        else:
            g = np.sum(g, axis=0)

    for axis, (gdim, tdim) in enumerate(zip(g.shape, shape)):
        if tdim == 1 and gdim != 1:
            if is_mlx:
                g = mx.sum(g, axis=axis, keepdims=True)
            else:
                g = np.sum(g, axis=axis, keepdims=True)

    try:
        return g.reshape(shape)
    except Exception:
        return g


def mul_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    def get_data(t):
        """Extract raw data from tensor or return as-is"""
        if isinstance(t, (int, float)):
            return t
        elif isinstance(t, ts.Tensor):
            return t.data
        else:
            return t

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        other_data = get_data(tensors[1])
        if mx and (isinstance(grad_data, mx.array) or isinstance(other_data, mx.array)):
            grad_mx = (
                grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
            )
            other_mx = (
                other_data
                if isinstance(other_data, mx.array)
                else (
                    mx.array(other_data)
                    if isinstance(other_data, np.ndarray)
                    else other_data
                )
            )
            tensors[0].grad = grad_mx * other_mx
        else:
            tensors[0].grad = grad_data * other_data
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        other_data = get_data(tensors[0])
        if mx and (isinstance(grad_data, mx.array) or isinstance(other_data, mx.array)):
            grad_mx = (
                grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
            )
            other_mx = (
                other_data
                if isinstance(other_data, mx.array)
                else (
                    mx.array(other_data)
                    if isinstance(other_data, np.ndarray)
                    else other_data
                )
            )
            tensors[1].grad = grad_mx * other_mx
        else:
            tensors[1].grad = grad_data * other_data
        tensors[1].backward(tensors[1].grad)
    else:
        return grad_data * get_data(tensors[1])
    return grad_data * get_data(tensors[0])


def add_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        grad0 = (
            grad_outputs[0].data
            if isinstance(grad_outputs[0], ts.Tensor)
            else grad_outputs[0]
        )
        tensors[0].grad = _reduce_grad_to_shape(grad0, tensors[0].data.shape)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        grad1 = (
            grad_outputs[0].data
            if isinstance(grad_outputs[0], ts.Tensor)
            else grad_outputs[0]
        )
        tensors[1].grad = _reduce_grad_to_shape(grad1, tensors[1].data.shape)
        tensors[1].backward(tensors[1].grad)
    else:
        return grad_outputs[0]
    return grad_outputs[0]


def sub_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        grad0 = (
            grad_outputs[0].data
            if isinstance(grad_outputs[0], ts.Tensor)
            else grad_outputs[0]
        )
        tensors[0].grad = _reduce_grad_to_shape(grad0, tensors[0].data.shape)
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        grad1 = (
            grad_outputs[0].data
            if isinstance(grad_outputs[0], ts.Tensor)
            else grad_outputs[0]
        )
        tensors[1].grad = _reduce_grad_to_shape(-grad1, tensors[1].data.shape)
        tensors[1].backward(tensors[1].grad)
    else:
        return -grad_outputs[0]
    return grad_outputs[0]


def neg_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = (
            -grad_outputs[0]
            if isinstance(grad_outputs[0], (int, float))
            else -grad_outputs[0].data
        )
        tensors[0].backward(tensors[0].grad)
    else:
        return (
            -grad_outputs[0]
            if isinstance(grad_outputs[0], (int, float))
            else -grad_outputs[0].data
        )
    return grad_outputs[0]


def truediv_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    is_reverse_div = isinstance(tensors[0], (int, float)) and isinstance(
        tensors[1], ts.Tensor
    )

    if is_reverse_div:
        if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
            tensors[1].grad = -grad_data * tensors[0] / (tensors[1].data ** 2)
            tensors[1].backward(tensors[1].grad)
        return grad_data

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data / (
            tensors[1] if isinstance(tensors[1], (int, float)) else tensors[1].data
        )
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        tensors[1].grad = -grad_data * tensors[0].data / (tensors[1].data ** 2)
        tensors[1].backward(tensors[1].grad)
    return grad_data


def matmul_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    def ensure_compatible(a, b):
        if mx and isinstance(a, mx.array) and not isinstance(b, mx.array):
            return a, mx.array(b)
        elif mx and isinstance(b, mx.array) and not isinstance(a, mx.array):
            return mx.array(a), b
        return a, b

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        grad_compat, weight_T = ensure_compatible(grad_data, tensors[1].data.T)
        tensors[0].grad = grad_compat @ weight_T
        tensors[0].backward(tensors[0].grad)
    if isinstance(tensors[1], ts.Tensor) and tensors[1].requires_grad:
        input_T, grad_compat = ensure_compatible(tensors[0].data.T, grad_data)
        tensors[1].grad = input_T @ grad_compat
        tensors[1].backward(tensors[1].grad)

    grad_compat, weight_T = ensure_compatible(grad_data, tensors[1].data.T)
    return grad_compat @ weight_T


def pow_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * tensors[1] * (tensors[0].data ** (tensors[1] - 1))
        tensors[0].backward(tensors[0].grad)
    return grad_data * tensors[1] * (tensors[0].data ** (tensors[1] - 1))


def max_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * (tensors[0].data > tensors[1])
        tensors[0].backward(tensors[0].grad)
    return grad_data * (tensors[0].data > tensors[1])


def sum_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        shape = tensors[0].data.shape
        if mx and isinstance(tensors[0].data, mx.array):
            g = grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
            tensors[0].grad = mx.broadcast_to(g, shape)
        else:
            if not isinstance(grad_data, np.ndarray):
                grad_data = np.array(grad_data)
            tensors[0].grad = np.broadcast_to(grad_data, shape).copy()
        tensors[0].backward(tensors[0].grad)
    return grad_data


def mean_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        n = tensors[0].data.size
        shape = tensors[0].data.shape
        if mx and isinstance(tensors[0].data, mx.array):
            g = grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
            tensors[0].grad = mx.broadcast_to(g / n, shape)
        else:
            if not isinstance(grad_data, np.ndarray):
                grad_data = np.array(grad_data)
            tensors[0].grad = np.broadcast_to(grad_data / n, shape).copy()
        tensors[0].backward(tensors[0].grad)
    return grad_data / tensors[0].data.size


def sigmoid_backward(tensors: Any, grad_outputs: Any) -> Any:
    """
    Backward for sigmoid(x).

    `tensors` is expected to be [x, y] where y = sigmoid(x) (raw array).
    """
    x = tensors[0]
    y = tensors[1]
    grad = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    # dy/dx = y * (1 - y)
    if mx and (isinstance(grad, mx.array) or isinstance(y, mx.array)):
        grad_mx = grad if isinstance(grad, mx.array) else mx.array(grad)
        y_mx = y if isinstance(y, mx.array) else mx.array(y)
        grad_x = grad_mx * (y_mx * (1 - y_mx))
    else:
        grad_np = np.array(grad) if not isinstance(grad, np.ndarray) else grad
        y_np = np.array(y) if not isinstance(y, np.ndarray) else y
        grad_x = grad_np * (y_np * (1 - y_np))

    if isinstance(x, ts.Tensor) and x.requires_grad:
        x.grad = grad_x
        x.backward(x.grad)

    return grad_x


def as_strided_backward(tensors: Any, grad_outputs: Any) -> Any:
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = np.lib.stride_tricks.as_strided(
            grad_outputs[0],
            shape=tensors[0].data.shape,
            strides=tensors[1].data.strides,
            storage_offset=tensors[2].data,
        )
        tensors[0].backward(tensors[0].grad)
    return grad_outputs[0]


def trace_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        n = min(tensors[0].data.shape[-2:])
        if mx and isinstance(tensors[0].data, mx.array):
            tensors[0].grad = mx.eye(n) * grad_data
        else:
            tensors[0].grad = np.eye(n) * np.asarray(grad_data)
        tensors[0].backward(tensors[0].grad)
    return grad_data


def conv1d_backward(
    tensors: Any, grad_outputs: Any, stride: int = 1, padding: int = 0
) -> Any:
    x, weight, bias = tensors
    grad_output = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    if not isinstance(grad_output, np.ndarray):
        grad_output = np.array(grad_output)

    weight_data = weight.data if isinstance(weight, ts.Tensor) else weight

    if isinstance(x, ts.Tensor) and x.requires_grad:
        grad_x = conv1d_grad_input(grad_output, weight_data, stride, padding)
        if grad_x is not None:
            x.grad = grad_x
            x.backward(x.grad)

    if isinstance(weight, ts.Tensor) and weight.requires_grad:
        grad_weight = conv1d_grad_weight(grad_output, x, stride, padding)
        if grad_weight is not None:
            weight.grad = grad_weight
            weight.backward(weight.grad)

    if bias is not None and isinstance(bias, ts.Tensor) and bias.requires_grad:
        grad_bias = grad_output.sum(axis=(0, 2))
        bias.grad = grad_bias
        bias.backward(bias.grad)

    return grad_output


def conv1d_grad_input(
    grad_output: np.ndarray, weight: "ts.Tensor", stride: int = 1, padding: int = 0
) -> np.ndarray:
    batch_size, out_channels, out_length = grad_output.shape
    _, in_channels, kernel_size = weight.data.shape

    length = (out_length - 1) * stride + kernel_size - 2 * padding

    grad_input_padded = np.zeros((batch_size, in_channels, length + 2 * padding))

    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_length):
                il_start = ol * stride

                for k in range(kernel_size):
                    grad_input_padded[b, :, il_start + k] += (
                        weight[oc, :, k] * grad_output[b, oc, ol]
                    )

    if padding > 0:
        return grad_input_padded[:, :, padding:-padding]
    return grad_input_padded


def conv1d_grad_weight(
    grad_output: np.ndarray, x: "ts.Tensor", stride: int = 1, padding: int = 0
) -> np.ndarray:
    batch_size, out_channels, out_length = grad_output.shape
    _, in_channels, _ = x.data.shape

    kernel_size = x.data.shape[2] - (out_length - 1) * stride + 2 * padding

    grad_weight = np.zeros((out_channels, in_channels, kernel_size))

    if padding > 0:
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    else:
        x_padded = x.data

    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_length):
                il_start = ol * stride
                il_end = il_start + kernel_size

                x_window = x_padded[b, :, il_start:il_end]

                grad_weight[oc] += x_window * grad_output[b, oc, ol]

    return grad_weight


def reshape_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        if mx and isinstance(grad_data, mx.array):
            tensors[0].grad = mx.reshape(grad_data, tensors[0].data.shape)
        else:
            if not isinstance(grad_data, np.ndarray):
                grad_data = np.array(grad_data)
            tensors[0].grad = grad_data.reshape(tensors[0].data.shape)
        tensors[0].backward(tensors[0].grad)

    if mx and isinstance(grad_data, mx.array):
        return mx.reshape(grad_data, tensors[0].data.shape)
    else:
        if not isinstance(grad_data, np.ndarray):
            grad_data = np.array(grad_data)
        return grad_data.reshape(tensors[0].data.shape)


def getitem_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        if mx and isinstance(tensors[0].data, mx.array):
            g = np.zeros(tensors[0].data.shape, dtype=np.float32)
            g[tensors[1]] = np.array(grad_data)
            tensors[0].grad = mx.array(g)
        else:
            tensors[0].grad = np.zeros_like(tensors[0].data)
            tensors[0].grad[tensors[1]] = grad_data
        tensors[0].backward(tensors[0].grad)
    if mx and isinstance(tensors[0].data, mx.array):
        return mx.zeros_like(tensors[0].data)
    return np.zeros_like(tensors[0].data)


def setitem_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        if tensors[0].grad is None:
            if mx and isinstance(tensors[0].data, mx.array):
                tensors[0].grad = mx.zeros_like(tensors[0].data)
            else:
                tensors[0].grad = np.zeros_like(tensors[0].data)
        if mx and isinstance(tensors[0].grad, mx.array):
            g = np.array(tensors[0].grad)
            g[tensors[1]] = np.array(grad_data)
            tensors[0].grad = mx.array(g)
        else:
            tensors[0].grad[tensors[1]] = grad_data
        tensors[0].backward(tensors[0].grad)
    if mx and isinstance(tensors[0].data, mx.array):
        return mx.zeros_like(tensors[0].data)
    return np.zeros_like(tensors[0].data)


def item_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data
        tensors[0].backward(tensors[0].grad)
    return grad_data


def transpose_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        if len(tensors) == 3:
            dim0, dim1 = tensors[1], tensors[2]
            if mx and isinstance(grad_data, mx.array):
                tensors[0].grad = mx.swapaxes(grad_data, dim0, dim1)
            else:
                tensors[0].grad = np.swapaxes(grad_data, dim0, dim1)
        else:
            if mx and isinstance(grad_data, mx.array):
                tensors[0].grad = mx.transpose(grad_data)
            else:
                tensors[0].grad = grad_data.T
        tensors[0].backward(tensors[0].grad)

    if len(tensors) == 3:
        dim0, dim1 = tensors[1], tensors[2]
        if mx and isinstance(grad_data, mx.array):
            return mx.swapaxes(grad_data, dim0, dim1)
        return np.swapaxes(grad_data, dim0, dim1)
    if mx and isinstance(grad_data, mx.array):
        return mx.transpose(grad_data)
    return grad_data.T


def tanh_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * (1 - tensors[0].data ** 2)
        tensors[0].backward(tensors[0].grad)
    return grad_data * (1 - tensors[0].data ** 2)


def softmax_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    x_data = tensors[0].data if isinstance(tensors[0], ts.Tensor) else tensors[0]

    if mx and isinstance(x_data, mx.array):
        s = mx.softmax(x_data, axis=-1)
        g = grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
        grad_x = s * (g - mx.sum(g * s, axis=-1, keepdims=True))
    else:
        if not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)
        e_x = np.exp(x_data - x_data.max(axis=-1, keepdims=True))
        s = e_x / e_x.sum(axis=-1, keepdims=True)
        g = grad_data if isinstance(grad_data, np.ndarray) else np.array(grad_data)
        grad_x = s * (g - (g * s).sum(axis=-1, keepdims=True))

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_x
        tensors[0].backward(tensors[0].grad)
    return grad_x


def gelu_backward(tensors: Any, grad_outputs: Any) -> Any:
    """
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    d(GELU)/dx = 0.5 * (1 + tanh(t)) + 0.5 * x * (1 − tanh(t)²) * t'
    where  t  = sqrt(2/π) * (x + 0.044715 * x³)
    and    t' = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
    """
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    x_data = tensors[0].data if isinstance(tensors[0], ts.Tensor) else tensors[0]

    if mx and isinstance(x_data, mx.array):
        g = grad_data if isinstance(grad_data, mx.array) else mx.array(grad_data)
        c = mx.array(np.sqrt(2.0 / np.pi))
        t = c * (x_data + 0.044715 * mx.power(x_data, 3))
        tanh_t = mx.tanh(t)
        t_prime = c * (1.0 + 3.0 * 0.044715 * x_data**2)
        grad_x = g * (0.5 * (1.0 + tanh_t) + 0.5 * x_data * (1.0 - tanh_t**2) * t_prime)
    else:
        x_np = x_data if isinstance(x_data, np.ndarray) else np.array(x_data)
        g = grad_data if isinstance(grad_data, np.ndarray) else np.array(grad_data)
        c = np.sqrt(2.0 / np.pi)
        t = c * (x_np + 0.044715 * np.power(x_np, 3))
        tanh_t = np.tanh(t)
        t_prime = c * (1.0 + 3.0 * 0.044715 * x_np**2)
        grad_x = g * (0.5 * (1.0 + tanh_t) + 0.5 * x_np * (1.0 - tanh_t**2) * t_prime)

    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_x
        tensors[0].backward(tensors[0].grad)
    return grad_x


def dropout_backward(tensors: Any, grad_outputs: Any) -> Any:
    """
    tensors = [x, mask]
    mask is the scaled binary mask produced during the forward pass
    (already multiplied by 1 / (1 - p)).
    Backward is simply grad * mask — same operation as the forward, no new randomness.
    """
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    x = tensors[0]
    mask = tensors[1]  # raw numpy array

    if mx and isinstance(grad_data, mx.array):
        m = mx.array(mask) if not isinstance(mask, mx.array) else mask
        grad_x = grad_data * m
    else:
        g = grad_data if isinstance(grad_data, np.ndarray) else np.array(grad_data)
        grad_x = g * mask

    if isinstance(x, ts.Tensor) and x.requires_grad:
        x.grad = grad_x
        x.backward(x.grad)
    return grad_x


def cross_entropy_backward(tensors: Any, grad_outputs: Any) -> Any:
    """
    Cross-entropy from raw logits.
    tensors = [logits_tensor, target_tensor]
      logits : (B, C)  raw (pre-softmax) scores
      target : (B,)    integer class indices   OR   (B, C) one-hot / soft targets

    Gradient of mean cross-entropy w.r.t. logits:
        (softmax(logits) − one_hot(target)) / B
    """
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    logits = tensors[0]
    target = tensors[1]

    logits_data = logits.data if isinstance(logits, ts.Tensor) else logits
    target_data = target.data if isinstance(target, ts.Tensor) else target

    logits_np = np.array(logits_data).astype(np.float32)
    target_np = np.array(target_data)

    B = logits_np.shape[0]
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    softmax_probs = np.exp(shifted)
    softmax_probs /= softmax_probs.sum(axis=1, keepdims=True)

    grad_logits = softmax_probs.copy()
    if target_np.ndim == 1:
        grad_logits[np.arange(B), target_np.astype(int)] -= 1.0
    else:
        grad_logits -= target_np.astype(np.float32)
    grad_logits /= B

    # Scale by the upstream scalar gradient (typically 1.0 for a scalar loss)
    upstream = float(np.array(grad_data).flat[0])
    grad_logits = (grad_logits * upstream).astype(np.float32)

    if mx and isinstance(logits_data, mx.array):
        grad_arr = mx.array(grad_logits)
    else:
        grad_arr = grad_logits

    if isinstance(logits, ts.Tensor) and logits.requires_grad:
        logits.grad = grad_arr
        logits.backward(logits.grad)
    return grad_arr


def squeeze_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data
        tensors[0].backward(tensors[0].grad)
    return grad_data


def exp_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * tensors[0].data
        tensors[0].backward(tensors[0].grad)
    return grad_data * tensors[0].data


def log_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data / tensors[0].data
        tensors[0].backward(tensors[0].grad)
    return grad_data / tensors[0].data


def clamp_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = (
            grad_data
            * (tensors[0].data >= tensors[1])
            * (tensors[0].data <= tensors[2])
        )
        tensors[0].backward(tensors[0].grad)
    return grad_data * (tensors[0].data >= tensors[1]) * (tensors[0].data <= tensors[2])


def minimum_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * (tensors[0].data <= tensors[1])
        tensors[0].backward(tensors[0].grad)
    return grad_data * (tensors[0].data <= tensors[1])


def maximum_backward(tensors: Any, grad_outputs: Any) -> Any:
    grad_data = (
        grad_outputs[0].data
        if isinstance(grad_outputs[0], ts.Tensor)
        else grad_outputs[0]
    )
    if isinstance(tensors[0], ts.Tensor) and tensors[0].requires_grad:
        tensors[0].grad = grad_data * (tensors[0].data >= tensors[1])
        tensors[0].backward(tensors[0].grad)
    return grad_data * (tensors[0].data >= tensors[1])
