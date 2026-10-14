from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer.autograd as autograd
import numpy as np
from energizer._mlx import mx


class MSELoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.reduction = reduction

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        diff = (input - target) ** 2
        if self.reduction == "sum":
            return diff.sum()
        return diff.mean()


def _is_mlx(a) -> bool:
    return mx is not None and isinstance(a, mx.array)


def _ce_forward_mlx(logits_data, target_data, reduction):
    """Cross-entropy on MLX arrays; stays lazy, no host transfer."""
    logits = logits_data.astype(mx.float32)
    target = target_data if _is_mlx(target_data) else mx.array(np.asarray(target_data))
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    if target.ndim == 1:
        idx = target.astype(mx.int32)[:, None]
        nll = -mx.take_along_axis(log_probs, idx, axis=1)[:, 0]
    else:
        nll = -(target.astype(mx.float32) * log_probs).sum(axis=1)
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    return nll


def _ce_backward_mlx(logits_data, target_data, reduction, grad):
    logits = logits_data.astype(mx.float32)
    target = target_data if _is_mlx(target_data) else mx.array(np.asarray(target_data))
    probs = mx.softmax(logits, axis=1)
    if target.ndim == 1:
        classes = mx.arange(logits.shape[1])[None, :]
        onehot = (classes == target.astype(mx.int32)[:, None]).astype(mx.float32)
    else:
        onehot = target.astype(mx.float32)
    g = probs - onehot
    if reduction == "mean":
        g = g / logits.shape[0]
    up = grad if _is_mlx(grad) else mx.array(np.asarray(grad, dtype=np.float32))
    if up.size == 1:
        up = up.reshape(())
        return g * up
    return g * up[:, None]  # reduction="none": per-sample upstream gradient


class CrossEntropyFn(autograd.Function):
    @staticmethod
    def forward(ctx, logits_data, target_data, reduction):
        ctx.save_for_backward(logits_data, target_data, reduction)

        if _is_mlx(logits_data):
            return _ce_forward_mlx(logits_data, target_data, reduction)

        logits_np = np.array(logits_data).astype(np.float32)
        target_np = np.array(target_data)

        B = logits_np.shape[0]

        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

        if target_np.ndim == 1:
            nll = -log_probs[np.arange(B), target_np.astype(int)]
        else:
            nll = -(target_np.astype(np.float32) * log_probs).sum(axis=1)

        if reduction == "sum":
            loss_val = float(nll.sum())
        elif reduction == "mean":
            loss_val = float(nll.mean())
        else:
            loss_val = nll

        if isinstance(loss_val, float):
            return np.array(loss_val, dtype=np.float32)
        return loss_val.astype(np.float32)

    @staticmethod
    def backward(ctx, grad):
        logits_data, target_data, reduction = ctx.saved_tensors

        if _is_mlx(logits_data):
            return (
                _ce_backward_mlx(logits_data, target_data, reduction, grad),
                None,
                None,
            )

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

        if reduction == "mean":
            grad_logits /= B

        upstream = float(np.array(grad).flat[0])
        grad_logits = (grad_logits * upstream).astype(np.float32)

        return (grad_logits, None, None)


class CrossEntropyLoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.reduction = reduction

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return CrossEntropyFn.apply(input, target, self.reduction)
