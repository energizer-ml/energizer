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


class CrossEntropyFn(autograd.Function):
    @staticmethod
    def forward(ctx, logits_data, target_data, reduction):
        ctx.save_for_backward(logits_data, target_data, reduction)

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
