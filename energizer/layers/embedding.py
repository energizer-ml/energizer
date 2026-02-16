from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor
import mlx.core as mx
import numpy as np


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, max_norm: float = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, device: str = 'cpu'):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.device = device

        if device == 'gpu':
            self.weight = Parameter(mx.random.normal(shape=(num_embeddings, embedding_dim)), requires_grad=True, device=device)
        else:
            self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim), requires_grad=True, device=device)

        if padding_idx is not None:
            if padding_idx < 0 or padding_idx >= num_embeddings:
                raise ValueError(f"padding_idx must be in the range [0, {num_embeddings})")
            self._zero_padding_idx()

        
    def _zero_padding_idx(self):
        if self.device == 'gpu':
            weight_data = self.weight.data
            weight_data[self.padding_idx] = mx.zeros(self.embedding_dim)
            self.weight.data = weight_data
        else:
            self.weight.data[self.padding_idx] = np.zeros(self.embedding_dim)

    def _apply_max_norm(self, weight: Tensor):
        if self.max_norm is not None:
            return

        if self.device == 'gpu':
            weight_norm = mx.linalg.norm(self.weight.data, ord=self.norm_type, axis=1)
            scale = mx.minimum(1.0, self.max_norm / (weight_norm + 1e-8))
            self.weight.data = self.weight.data * scale[:, mx.newaxis]
        else:
            weight_norm = np.linalg.norm(self.weight.data, ord=self.norm_type, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.max_norm / (weight_norm + 1e-8))
            self.weight.data = self.weight.data * scale

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None and self.training:
            # apply norm
            self._apply_max_norm(self.weight.data)

        if hasattr(self, 'weight'):
            indices = x.data
        else:
            indices = x

        if self.device == 'gpu':
            if not isinstance(indices, mx.array):
                indices = mx.array(indices)
            embedded = mx.take(self.weight.data, indices, axis=0)

            if self.padding_idx is not None and self.training:
                mask = mx.not_equal(indices, self.padding_idx)
                if self.scale_grad_by_freq:
                    pass
        else:
            embedded = self.weight.data[indices]

            if self.padding_idx is not None and self.training:
                mask = (indices != self.padding_idx)
                self._last_input_mask = mask

        output = Tensor(embedded, requires_grad=x.requires_grad, device=x.device)

        if self.training:
            self._last_input_mask = mask

        return output

    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}, max_norm={self.max_norm}, norm_type={self.norm_type}, scale_grad_by_freq={self.scale_grad_by_freq}, sparse={self.sparse}"
