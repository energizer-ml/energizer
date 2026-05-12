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
from .launcher import ParallelLaunchConfig, build_mlx_launch_command, launch

__all__ = [
    "AllToShardedLinear",
    "DeviceGroup",
    "ParallelLaunchConfig",
    "ShardedTensor",
    "ShardedToAllLinear",
    "all_sum",
    "average_gradients",
    "build_mlx_launch_command",
    "data_parallel_step",
    "gather",
    "init_device_group",
    "launch",
    "shard",
    "shard_inplace",
    "shard_linear",
    "split_batch",
]
