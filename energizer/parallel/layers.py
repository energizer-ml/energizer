from __future__ import annotations
from energizer.backend import normalize_device

from typing import Literal

import numpy as np

from energizer._mlx import mx
from energizer.layers.linear import Linear
from energizer.neural_network import Module, Parameter
from energizer.tensor import Tensor

from .core import DeviceGroup, ShardedTensor, all_sum, raw_split

ShardType = Literal["all-to-sharded", "sharded-to-all"]


def _init_weight(rows: int, cols: int, device: str):
    device = normalize_device(device)
    if device == "mlx":
        if mx is None:
            raise RuntimeError("Cannot initialize MLX sharded layer without MLX")
        return mx.random.normal(shape=(rows, cols)) * 0.01
    return np.random.randn(rows, cols).astype(np.float32) * 0.01


def _init_bias(size: int, device: str):
    device = normalize_device(device)
    if device == "mlx":
        if mx is None:
            raise RuntimeError("Cannot initialize MLX sharded layer without MLX")
        return mx.zeros(size)
    return np.zeros(size, dtype=np.float32)


def _partition_size(total: int, parts: int, rank: int) -> int:
    base, extra = divmod(total, parts)
    return base + (1 if rank < extra else 0)


class _ShardedLinearBase(Module):
    def _register_shard(self, name: str, param: Parameter):
        self.add_parameter(name, param)
        return param

    def parameters(self):
        return super().parameters()


class AllToShardedLinear(_ShardedLinearBase):
    """
    Linear layer with output-feature sharding.

    Each shard receives the full input and computes only its output columns.
    The output is intentionally left sharded so it can feed directly into
    ``ShardedToAllLinear`` without an intermediate gather.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group: DeviceGroup | None = None,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.group = group or DeviceGroup.local(1)
        self.use_bias = bias

        ranks = (
            [self.group.rank] if self.group.is_distributed else range(self.group.size)
        )
        self.weight_shards = []
        self.bias_shards = []
        for rank in ranks:
            rows = _partition_size(out_features, self.group.size, rank)
            weight = Parameter(
                _init_weight(rows, in_features, device),
                requires_grad=True,
                device=device,
            )
            self.weight_shards.append(self._register_shard(f"weight_{rank}", weight))
            if bias:
                bias_param = Parameter(
                    _init_bias(rows, device),
                    requires_grad=True,
                    device=device,
                )
                self.bias_shards.append(
                    self._register_shard(f"bias_{rank}", bias_param)
                )

    def forward(self, x: Tensor) -> ShardedTensor:
        outputs = []
        for idx, weight in enumerate(self.weight_shards):
            out = x @ weight.T
            if self.use_bias:
                out = out + self.bias_shards[idx]
            outputs.append(out)
        return ShardedTensor(
            outputs,
            axis=-1,
            group=self.group,
            global_shape=(*x.shape[:-1], self.out_features),
        )

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.use_bias}, shards={self.group.size}"
        )


class ShardedToAllLinear(_ShardedLinearBase):
    """
    Linear layer with input-feature sharding.

    The input must already be sharded along its last dimension. Each shard
    computes a partial output and the partial outputs are summed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group: DeviceGroup | None = None,
        device: str = "cpu",
    ):
        device = normalize_device(device)
        super().__init__(device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.group = group or DeviceGroup.local(1)
        self.use_bias = bias

        ranks = (
            [self.group.rank] if self.group.is_distributed else range(self.group.size)
        )
        self.weight_shards = []
        for rank in ranks:
            cols = _partition_size(in_features, self.group.size, rank)
            weight = Parameter(
                _init_weight(out_features, cols, device),
                requires_grad=True,
                device=device,
            )
            self.weight_shards.append(self._register_shard(f"weight_{rank}", weight))

        self.bias = None
        if bias:
            self.bias = self._register_shard(
                "bias",
                Parameter(
                    _init_bias(out_features, device),
                    requires_grad=True,
                    device=device,
                ),
            )

    def forward(self, x: ShardedTensor | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x = ShardedTensor(
                [x],
                axis=-1,
                group=DeviceGroup.local(1),
                global_shape=x.shape,
            )

        if len(x.shards) != len(self.weight_shards):
            if self.group.is_distributed and len(self.weight_shards) == 1:
                input_shards = [x.local]
            else:
                raise ValueError(
                    "ShardedToAllLinear expected the same number of input and "
                    f"weight shards, got {len(x.shards)} and "
                    f"{len(self.weight_shards)}"
                )
        else:
            input_shards = list(x.shards)

        partials = [
            input_shard @ weight.T
            for input_shard, weight in zip(input_shards, self.weight_shards)
        ]
        out = all_sum(
            partials[0] if self.group.is_distributed else partials,
            self.group,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.use_bias}, shards={self.group.size}"
        )


def shard_linear(
    layer: Linear,
    mode: ShardType,
    group: DeviceGroup | None = None,
) -> AllToShardedLinear | ShardedToAllLinear:
    """
    Convert an Energizer Linear layer into a tensor-parallel Linear layer.

    The original layer is not modified. Weights and bias are copied into the
    new layer shards.
    """

    group = group or DeviceGroup.local(1)
    bias = layer.bias is not None
    if mode == "all-to-sharded":
        sharded = AllToShardedLinear(
            layer.in_features,
            layer.out_features,
            bias=bias,
            group=group,
            device=layer.device,
        )
        weight_parts = raw_split(
            layer.weight.data,
            group.size,
            axis=0,
            device=layer.device,
        )
        bias_parts = (
            raw_split(
                layer.bias.data,
                group.size,
                axis=0,
                device=layer.device,
            )
            if bias
            else []
        )
        for idx, weight in enumerate(sharded.weight_shards):
            rank = group.rank if group.is_distributed else idx
            weight.data = weight_parts[rank]
        for idx, bias_param in enumerate(sharded.bias_shards):
            rank = group.rank if group.is_distributed else idx
            bias_param.data = bias_parts[rank]
        return sharded

    if mode == "sharded-to-all":
        sharded = ShardedToAllLinear(
            layer.in_features,
            layer.out_features,
            bias=bias,
            group=group,
            device=layer.device,
        )
        weight_parts = raw_split(
            layer.weight.data,
            group.size,
            axis=1,
            device=layer.device,
        )
        for idx, weight in enumerate(sharded.weight_shards):
            rank = group.rank if group.is_distributed else idx
            weight.data = weight_parts[rank]
        if bias:
            sharded.bias.data = layer.bias.data
        return sharded

    raise ValueError(
        "mode must be either 'all-to-sharded' or 'sharded-to-all', " f"got {mode!r}"
    )


def shard_inplace(layer: Linear, mode: ShardType, group: DeviceGroup | None = None):
    """
    Shard a Linear layer's parameters in-place for the current rank.

    This does not add communication to ``layer.forward``. Use it when a custom
    module handles the communication pattern itself.
    """

    group = group or DeviceGroup.local(1)
    rank = group.rank
    if mode == "all-to-sharded":
        layer.weight.data = raw_split(
            layer.weight.data,
            group.size,
            axis=0,
            device=layer.device,
        )[rank]
        layer.out_features = layer.weight.data.shape[0]
        if layer.bias is not None:
            layer.bias.data = raw_split(
                layer.bias.data,
                group.size,
                axis=0,
                device=layer.device,
            )[rank]
        return layer
    if mode == "sharded-to-all":
        layer.weight.data = raw_split(
            layer.weight.data,
            group.size,
            axis=1,
            device=layer.device,
        )[rank]
        layer.in_features = layer.weight.data.shape[1]
        return layer
    raise ValueError(
        "mode must be either 'all-to-sharded' or 'sharded-to-all', " f"got {mode!r}"
    )
