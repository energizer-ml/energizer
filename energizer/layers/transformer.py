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
        batch_size, seq_len, embed_dim = x.shape

        if self.device == "gpu":
            x_reshaped = mx.reshape(
                x.data, (batch_size, seq_len, self.num_heads, self.head_dim)
            )
            x_reshaped = mx.transpose(x_reshaped, (0, 2, 1, 3))
        else:
            x_reshaped = x.data.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            x_reshaped = x_reshaped.transpose(0, 2, 1, 3)

        return Tensor(x_reshaped, requires_grad=x.requires_grad, device=x.device)

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
    ) -> Tensor:
        if self.device == "gpu":
            scores = mx.matmul(q.data, mx.transpose(k.data, (0, 1, 3, 2)))
        else:
            scores = np.matmul(q.data, k.data.transpose(0, 1, 3, 2))

        scores = scores / (self.head_dim**0.5)

        if attn_mask is not None:
            scores = scores + attn_mask.data

        if key_padding_mask is not None:
            if self.device == "gpu":
                mask = key_padding_mask.data[:, None, None, :]
                scores = mx.where(mask, mx.full_like(scores, -float("inf")), scores)
            else:
                mask = key_padding_mask.data[:, None, None, :]
                scores = np.where(mask, -np.inf, scores)

        if self.device == "gpu":
            attn_weights = mx.softmax(scores, axis=-1)
            attn_weights = self.dropout_layer(Tensor(attn_weights, device="gpu")).data
            if not isinstance(attn_weights, mx.array):
                attn_weights = mx.array(attn_weights)
        else:
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            attn_weights = self.dropout_layer(Tensor(attn_weights)).data

        if self.device == "gpu":
            output = mx.matmul(attn_weights, v.data)
        else:
            output = np.matmul(attn_weights, v.data)

        batch_size, num_heads, seq_len, head_dim = output.shape

        if self.device == "gpu":
            output = mx.transpose(output, (0, 2, 1, 3))
            output = mx.reshape(output, (batch_size, seq_len, self.embed_dim))
        else:
            output = output.transpose(0, 2, 1, 3)
            output = output.reshape(batch_size, seq_len, self.embed_dim)

        return Tensor(output, requires_grad=q.requires_grad, device=q.device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
    ) -> Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        attn_output = self._scaled_dot_product_attention(
            q, k, v, attn_mask, key_padding_mask
        )

        output = self.out_proj(attn_output)

        return output


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
