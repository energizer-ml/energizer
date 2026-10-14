from energizer.backend import normalize_device
from typing import Callable, Union
from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer
import numpy as np
from energizer._mlx import mx


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.bias = bias

        self.head_dim = embed_dim // num_heads
        self.q_proj = energizer.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.k_proj = energizer.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.v_proj = energizer.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.out_proj = energizer.Linear(embed_dim, embed_dim, bias=bias, device=device)

        self.dropout_layer = energizer.Dropout(dropout)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def _additive_mask(self, mask: Tensor, like: Tensor, key_padding: bool) -> Tensor:
        """Boolean masks (True = masked) become additive -inf; float masks pass."""
        data = mask.data
        xp = mx if like.device == "mlx" else np
        if key_padding:
            data = data[:, None, None, :]
        if data.dtype == bool or str(data.dtype) == "bool":
            data = xp.where(data, -float("inf"), 0.0).astype(xp.float32)
        return Tensor(data, device=like.device)

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
    ) -> Tensor:
        # Built from differentiable Tensor ops, so autograd (and the MLX path)
        # needs no hand-written attention backward.
        scores = (q @ k.permute(0, 1, 3, 2)) * (1.0 / (self.head_dim**0.5))
        if attn_mask is not None:
            scores = scores + self._additive_mask(attn_mask, scores, False)
        if key_padding_mask is not None:
            scores = scores + self._additive_mask(key_padding_mask, scores, True)

        weights = self.dropout_layer(scores.softmax())
        out = (weights @ v).permute(0, 2, 1, 3)
        batch_size, seq_len, _, _ = out.shape
        return out.reshape((batch_size, seq_len, self.embed_dim))

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
    ) -> Tensor:
        q = self._reshape_for_attention(self.q_proj(query))
        k = self._reshape_for_attention(self.k_proj(key))
        v = self._reshape_for_attention(self.v_proj(value))
        attn_output = self._scaled_dot_product_attention(
            q, k, v, attn_mask, key_padding_mask
        )
        return self.out_proj(attn_output)


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable, Tensor] = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.dropout_p = dropout
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, device=device
        )

        self.linear1 = energizer.Linear(
            d_model, dim_feedforward, bias=bias, device=device
        )
        self.linear2 = energizer.Linear(
            dim_feedforward, d_model, bias=bias, device=device
        )

        self.dropout = energizer.Dropout(dropout)
        self.dropout1 = energizer.Dropout(dropout)
        self.dropout2 = energizer.Dropout(dropout)

        self.norm1 = energizer.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm2 = energizer.LayerNorm(d_model, eps=layer_norm_eps, device=device)

        if activation == "relu":
            self.activation = energizer.ReLU()
        elif activation == "gelu":
            self.activation = energizer.GELU()

    def _sa_block(
        self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None
    ) -> Tensor:
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout2(x)

    def forward(
        self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None
    ) -> Tensor:
        if not self.batch_first:
            src = src.transpose(0, 1)

        if self.norm_first:
            x = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x

    def extra_repr(self):
        return f"d_model={self.d_model}, nhead={self.nhead}, dim_feedforward={self.dim_feedforward}, dropout={self.dropout}, activation={self.activation.__class__.__name__}, layer_norm_eps={self.layer_norm_eps}, batch_first={self.batch_first}, norm_first={self.norm_first}, bias={self.bias}"


class TransformerEncoder(Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module = None,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.layers = energizer.ModuleList(
            [
                encoder_layer.__class__(
                    encoder_layer.d_model,
                    encoder_layer.nhead,
                    encoder_layer.linear1.out_features,
                    dropout=encoder_layer.dropout_p,
                    batch_first=encoder_layer.batch_first,
                    norm_first=encoder_layer.norm_first,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
