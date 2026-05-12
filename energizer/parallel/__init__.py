"""
Parallel training utilities for Energizer.

The module mirrors the tensor-parallel patterns used by MLX:

    - AllToShardedLinear shards a linear layer on its output features.
    - ShardedToAllLinear shards a linear layer on its input features and
      aggregates partial outputs.
    - DataParallel provides batch sharding and gradient averaging helpers.
"""

from .core import (
    DeviceGroup,
    ShardedTensor,
    all_sum,
    average_gradients,
    data_parallel_step,
    gather,
    init_device_group,
    shard,
    split_batch,
)
from .layers import AllToShardedLinear, ShardedToAllLinear, shard_inplace, shard_linear

__all__ = [
    "AllToShardedLinear",
    "DeviceGroup",
    "ShardedTensor",
    "ShardedToAllLinear",
    "all_sum",
    "average_gradients",
    "data_parallel_step",
    "gather",
    "init_device_group",
    "shard",
    "shard_inplace",
    "shard_linear",
    "split_batch",
]
