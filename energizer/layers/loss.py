from energizer.neural_network import Module
from energizer.tensor import Tensor
from energizer.function import Function
import energizer.derivatives as dv
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


class MSELoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = (input - target) ** 2
        if self.reduction == "sum":
            return diff.sum()
        return diff.mean()

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)


class CrossEntropyLoss(Module):
    def __init__(
        self, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        logits_np = np.array(input.data).astype(np.float32)
        target_np = np.array(target.data)

        B = logits_np.shape[0]

        shifted   = logits_np - logits_np.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

        if target_np.ndim == 1:
            nll = -log_probs[np.arange(B), target_np.astype(int)]
        else:
            nll = -(target_np.astype(np.float32) * log_probs).sum(axis=1)

        if self.reduction == "sum":
            loss_val = float(nll.sum())
        elif self.reduction == "mean":
            loss_val = float(nll.mean())
        else:
            loss_val = nll

        if isinstance(loss_val, float):
            if mx is not None and input.device == "gpu":
                loss_data = mx.array(loss_val)
            else:
                loss_data = np.array(loss_val, dtype=np.float32)
        else:
            loss_data = loss_val.astype(np.float32)

        return Tensor(
            loss_data,
            requires_grad=input.requires_grad,
            grad_fn=Function(dv.cross_entropy_backward, [input, target]),
            device=input.device,
        )

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)
