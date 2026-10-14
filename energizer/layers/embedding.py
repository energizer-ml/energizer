from energizer.autograd import Function
from energizer.backend import normalize_device
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import numpy as np
from energizer._mlx import mx


class EmbeddingFn(Function):
    """Row gather; backward scatter-adds into the weight (on-device for MLX)."""

    @staticmethod
    def forward(ctx, weight, indices, padding_idx):
        if ctx.device == "mlx":
            idx = indices if isinstance(indices, mx.array) else mx.array(indices)
            idx = idx.astype(mx.int32)
            out = mx.take(weight, idx, axis=0)
        else:
            idx = np.asarray(indices).astype(np.int64)
            out = weight[idx]
        ctx.save_for_backward(idx, weight.shape, padding_idx)
        return out

    @staticmethod
    def backward(ctx, grad):
        idx, shape, padding_idx = ctx.saved_tensors
        flat_idx = idx.reshape(-1)
        flat_grad = grad.reshape(-1, shape[1])
        if padding_idx is not None:
            keep = flat_idx != padding_idx
            flat_grad = flat_grad * keep[:, None].astype(flat_grad.dtype)
        if ctx.device == "mlx":
            gw = mx.zeros(shape).at[flat_idx].add(flat_grad)
        else:
            gw = np.zeros(shape, dtype=np.float32)
            np.add.at(gw, flat_idx, flat_grad)
        return gw, None, None


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.device = device

        if device == "mlx":
            self.weight = Parameter(
                mx.random.normal(shape=(num_embeddings, embedding_dim)),
                requires_grad=True,
                device=device,
            )
        else:
            self.weight = Parameter(
                np.random.randn(num_embeddings, embedding_dim),
                requires_grad=True,
                device=device,
            )

        if padding_idx is not None:
            if padding_idx < 0 or padding_idx >= num_embeddings:
                raise ValueError(
                    f"padding_idx must be in the range [0, {num_embeddings})"
                )
            self._zero_padding_idx()

    def _zero_padding_idx(self):
        if self.device == "mlx":
            weight_data = self.weight.data
            weight_data[self.padding_idx] = mx.zeros(self.embedding_dim)
            self.weight.data = weight_data
        else:
            self.weight.data[self.padding_idx] = np.zeros(self.embedding_dim)

    def _apply_max_norm(self, weight: Tensor):
        if self.max_norm is not None:
            return

        if self.device == "mlx":
            weight_norm = mx.linalg.norm(self.weight.data, ord=self.norm_type, axis=1)
            scale = mx.minimum(1.0, self.max_norm / (weight_norm + 1e-8))
            self.weight.data = self.weight.data * scale[:, mx.newaxis]
        else:
            weight_norm = np.linalg.norm(
                self.weight.data, ord=self.norm_type, axis=1, keepdims=True
            )
            scale = np.minimum(1.0, self.max_norm / (weight_norm + 1e-8))
            self.weight.data = self.weight.data * scale

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None and self.training:
            # apply norm
            self._apply_max_norm(self.weight.data)

        indices = x.data if isinstance(x, Tensor) else x
        return EmbeddingFn.apply(self.weight, indices, self.padding_idx)

    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}, max_norm={self.max_norm}, norm_type={self.norm_type}, scale_grad_by_freq={self.scale_grad_by_freq}, sparse={self.sparse}"
