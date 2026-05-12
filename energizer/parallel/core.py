from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

from energizer._mlx import MLX_AVAILABLE, mx
from energizer.autograd import Function
from energizer.backend import backend
from energizer.neural_network import Module
from energizer.tensor import Tensor


def _normalize_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def _split_ranges(size: int, parts: int) -> list[tuple[int, int]]:
    if parts < 1:
        raise ValueError("parts must be >= 1")
    base, extra = divmod(size, parts)
    ranges = []
    start = 0
    for idx in range(parts):
        width = base + (1 if idx < extra else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def _array_split(raw, parts: int, axis: int, device: str):
    _ = device
    axis = _normalize_axis(axis, len(raw.shape))
    ranges = _split_ranges(raw.shape[axis], parts)
    result = []
    for start, end in ranges:
        key = [slice(None)] * len(raw.shape)
        key[axis] = slice(start, end)
        result.append(raw[tuple(key)])
    return result


class _Concatenate(Function):
    @staticmethod
    def forward(ctx, *args):
        axis = args[-1]
        arrays = args[:-1]
        ctx.axis = axis
        ctx.sizes = [
            arr.shape[_normalize_axis(axis, len(arr.shape))]
            for arr in arrays
        ]
        lib = backend.lib(ctx.device)
        return lib.concatenate(arrays, axis=axis)

    @staticmethod
    def backward(ctx, grad):
        axis = _normalize_axis(ctx.axis, len(grad.shape))
        grads = []
        start = 0
        for size in ctx.sizes:
            end = start + size
            key = [slice(None)] * len(grad.shape)
            key[axis] = slice(start, end)
            grads.append(grad[tuple(key)])
            start = end
        grads.append(None)
        return tuple(grads)


class _DistributedAllSum(Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return _distributed_all_sum(value, group)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@dataclass(frozen=True)
class DeviceGroup:
    """
    Small wrapper around an optional MLX distributed group.

    For tests and local development, ``backend="local"`` simulates multiple
    shards in one process. For real multi-process MLX runs, construct the group
    with ``init_device_group(distributed=True)`` and launch with ``mlx.launch``.
    """

    size: int = 1
    rank: int = 0
    backend: str = "local"
    raw_group: object | None = None

    @classmethod
    def local(cls, size: int = 1, rank: int = 0) -> "DeviceGroup":
        if size < 1:
            raise ValueError("DeviceGroup size must be >= 1")
        if rank < 0 or rank >= size:
            raise ValueError("DeviceGroup rank must be in [0, size)")
        return cls(size=size, rank=rank)

    @property
    def is_distributed(self) -> bool:
        return self.raw_group is not None


def init_device_group(size: int | None = None, distributed: bool | None = None):
    """
    Initialize a local or MLX distributed device group.

    ``distributed=True`` calls ``mx.distributed.init()`` and uses the returned
    world size/rank. ``distributed=None`` auto-detects an MLX launch if
    ``mlx.core.distributed`` is available; otherwise it returns a local group.
    """

    if distributed is False or not MLX_AVAILABLE:
        return DeviceGroup.local(size or 1)

    if mx is None or not hasattr(mx, "distributed"):
        if distributed:
            raise RuntimeError("MLX distributed communication is not available")
        return DeviceGroup.local(size or 1)

    if distributed is None and size is not None and size <= 1:
        return DeviceGroup.local(size)

    try:
        world = mx.distributed.init()
    except Exception:
        if distributed:
            raise
        return DeviceGroup.local(size or 1)

    return DeviceGroup(
        size=world.size(),
        rank=world.rank(),
        backend="mlx",
        raw_group=world,
    )


class ShardedTensor:
    """
    Tensor container carrying shards and enough metadata to gather them.

    In local mode ``shards`` contains all partitions. In MLX distributed mode a
    process usually stores only its local shard, while the ``global_shape`` and
    ``axis`` describe the logical full tensor.
    """

    def __init__(
        self,
        shards: Sequence[Tensor],
        axis: int = -1,
        group: DeviceGroup | None = None,
        global_shape: tuple[int, ...] | None = None,
    ):
        if len(shards) == 0:
            raise ValueError("ShardedTensor requires at least one shard")
        self.shards = tuple(shards)
        self.axis = axis
        self.group = group or DeviceGroup.local(len(shards))
        self.global_shape = global_shape or self._infer_global_shape()

    def _infer_global_shape(self) -> tuple[int, ...]:
        shape = list(self.shards[0].shape)
        axis = _normalize_axis(self.axis, len(shape))
        shape[axis] = sum(shard.shape[axis] for shard in self.shards)
        return tuple(shape)

    @property
    def device(self) -> str:
        return self.shards[0].device

    @property
    def shape(self) -> tuple[int, ...]:
        return self.global_shape

    @property
    def local(self) -> Tensor:
        if len(self.shards) == 1:
            return self.shards[0]
        return self.shards[self.group.rank]

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.shards)

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, index: int) -> Tensor:
        return self.shards[index]

    def gather(self) -> Tensor:
        return gather(self)

    def to(self, device: str) -> "ShardedTensor":
        return ShardedTensor(
            [shard.to(device) for shard in self.shards],
            axis=self.axis,
            group=self.group,
            global_shape=self.global_shape,
        )

    def __repr__(self):
        return (
            f"ShardedTensor(shape={self.global_shape}, axis={self.axis}, "
            f"shards={len(self.shards)}, device='{self.device}')"
        )


def shard(tensor: Tensor, parts: int, axis: int = 0) -> ShardedTensor:
    """Split a Tensor into differentiable shards along ``axis``."""

    axis = _normalize_axis(axis, len(tensor.shape))
    ranges = _split_ranges(tensor.shape[axis], parts)
    shards = []
    for start, end in ranges:
        key = [slice(None)] * len(tensor.shape)
        key[axis] = slice(start, end)
        shards.append(tensor[tuple(key)])
    return ShardedTensor(
        shards,
        axis=axis,
        group=DeviceGroup.local(parts),
        global_shape=tensor.shape,
    )


def split_batch(tensor: Tensor, parts: int) -> ShardedTensor:
    """Shard a batch tensor along dimension 0 for data parallel work."""

    return shard(tensor, parts=parts, axis=0)


def gather(value: Tensor | ShardedTensor, axis: int | None = None) -> Tensor:
    """Gather local shards into one Tensor."""

    if isinstance(value, Tensor):
        return value
    if len(value.shards) == 1:
        return value.shards[0]
    gather_axis = value.axis if axis is None else axis
    return _Concatenate.apply(*value.shards, gather_axis)


def _distributed_all_sum(raw, group: DeviceGroup):
    if mx is None or group.raw_group is None:
        return raw
    try:
        return mx.distributed.all_sum(raw, group=group.raw_group)
    except TypeError:
        return mx.distributed.all_sum(raw)


def all_sum(value: Tensor | Sequence[Tensor], group: DeviceGroup | None = None):
    """
    Sum tensors across a group.

    Local mode sums the sequence passed by the caller. MLX distributed mode uses
    ``mx.distributed.all_sum`` for a single Tensor.
    """

    group = group or DeviceGroup.local(1)
    if isinstance(value, Tensor):
        if group.is_distributed and value.device == "gpu":
            return _DistributedAllSum.apply(value, group)
        return value

    values = list(value)
    if len(values) == 0:
        raise ValueError("all_sum requires at least one tensor")
    total = values[0]
    for tensor in values[1:]:
        total = total + tensor
    return total


def _replace_grad(param, grad_raw):
    param.grad = Tensor(grad_raw, device=param.device)


def average_gradients(module: Module, group: DeviceGroup | None = None) -> Module:
    """
    Average parameter gradients after backward for data-parallel training.

    In MLX distributed mode this performs an all-sum for each gradient and
    divides by the world size. In local mode it leaves gradients unchanged.
    """

    group = group or DeviceGroup.local(1)
    if group.size <= 1:
        return module

    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.data if isinstance(param.grad, Tensor) else param.grad
        if group.is_distributed and param.device == "gpu":
            grad = _distributed_all_sum(grad, group) / group.size
            _replace_grad(param, grad)
    return module


def data_parallel_step(
    module: Module,
    loss_fn: Callable[[Tensor], Tensor],
    batch: Tensor,
    optimizer,
    group: DeviceGroup | None = None,
):
    """
    Minimal data-parallel training step helper.

    The caller supplies a ``loss_fn`` that accepts one batch shard and returns a
    scalar loss. The helper runs backward, averages gradients, steps, and clears
    gradients. In local mode this is equivalent to a normal single-process step.
    """

    group = group or DeviceGroup.local(1)
    batch_shards = split_batch(batch, group.size)
    local_batch = batch_shards.local if group.size > 1 else batch
    loss = loss_fn(local_batch)
    loss.backward()
    average_gradients(module, group)
    optimizer.step()
    optimizer.zero_grad()
    return loss


def raw_split(data, parts: int, axis: int, device: str):
    return _array_split(data, parts=parts, axis=axis, device=device)


def raw_concat(arrays: Iterable, axis: int, device: str):
    lib = backend.lib(device)
    return lib.concatenate(tuple(arrays), axis=axis)
