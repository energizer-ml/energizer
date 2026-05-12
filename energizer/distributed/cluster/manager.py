"""
energizer/distributed/cluster/manager.py
─────────────────────────────────────────
Cluster — user-facing API for multi-machine training and inference.

This is the entry point for all distributed work in Energizer.
It wraps World (Phase 1), AllReduce/comm (Phase 2), and DataParallel
(Phase 3) behind a single clean interface.

Usage — two machines (MacBook Pro + Mac Mini):

    # ── On MacBook Pro (rank 0, coordinator) ──────────────────────────────
    from energizer.distributed.cluster import Cluster

    cluster = Cluster.coordinator(port=29500)
    cluster.add_device(rank=1, addr="192.168.1.20")

    cluster.connect()           # blocks until Mac Mini joins
    cluster.ping_all()          # verify latency

    cluster.fit(model, dataloader, epochs=10)

    cluster.close()

    # ── On Mac Mini (rank 1, worker) ──────────────────────────────────────
    from energizer.distributed.cluster import Cluster

    cluster = Cluster.worker(rank=1, coordinator_addr="192.168.1.5", port=29500)
    cluster.connect()           # connects to MacBook Pro

    cluster.fit(model, dataloader, epochs=10)   # same call, different shard

    cluster.close()
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Union

from .device import DeviceInfo, DeviceRole, DeviceStatus

logger = logging.getLogger("energizer.distributed.cluster")

class ClusterState:
    IDLE       = "idle"
    CONNECTING = "connecting"
    READY      = "ready"
    TRAINING   = "training"
    PREDICTING = "predicting"
    CLOSED     = "closed"


# ── Cluster ───────────────────────────────────────────────────────────────────

class Cluster:
    """
    Manages a group of Energizer machines for distributed training/inference.

    Responsibilities:
        - Register and track devices (DeviceInfo)
        - Orchestrate World rendezvous (connect / disconnect)
        - Expose fit() and predict() that dispatch work across devices
        - Health monitoring (ping_all, status)

    Attributes:
        rank          This process's rank.
        world_size    Total number of devices (set automatically as devices are added).
        devices       Ordered list of DeviceInfo, indexed by rank.
        state         Current ClusterState string.
    """

    def __init__(self, rank: int, addr: str, port: int = 29500):
        self.rank  = rank
        self.addr  = addr
        self.port  = port
        self.state = ClusterState.IDLE

        self.devices: list[DeviceInfo] = []

        self._world = None

        local = DeviceInfo.local(rank=rank, port=port)
        if rank == 0:
            local.addr = addr
        self._register(local)

    @classmethod
    def coordinator(
        cls,
        port: int = 29500,
        bind_addr: str = "0.0.0.0",
    ) -> "Cluster":
        logger.info(f"Creating coordinator cluster on port {port}")
        return cls(rank=0, addr=bind_addr, port=port)

    @classmethod
    def worker(
        cls,
        rank: int,
        coordinator_addr: str,
        port: int = 29500,
    ) -> "Cluster":
        if rank == 0:
            raise ValueError("rank=0 is reserved for the coordinator. Use Cluster.coordinator().")
        logger.info(f"Creating worker cluster rank={rank}, coordinator={coordinator_addr}:{port}")
        return cls(rank=rank, addr=coordinator_addr, port=port)

    @classmethod
    def single(cls) -> "Cluster":
        c = cls(rank=0, addr="0.0.0.0", port=29500)
        c.state = ClusterState.READY
        return c

    def _register(self, device: DeviceInfo) -> None:
        while len(self.devices) <= device.rank:
            self.devices.append(None)
        self.devices[device.rank] = device

    def add_device(
        self,
        rank: int,
        addr: str,
        port: Optional[int] = None,
        hostname: Optional[str] = None,
    ) -> "Cluster":
        if rank == 0:
            raise ValueError("rank=0 is this machine (coordinator). Cannot add it as remote.")
        device = DeviceInfo.remote(
            rank     = rank,
            addr     = addr,
            port     = port or self.port,
            hostname = hostname,
        )
        self._register(device)
        logger.info(f"Registered device: {device}")
        return self

    def remove_device(self, rank: int) -> "Cluster":
        if self.state != ClusterState.IDLE:
            raise RuntimeError("Cannot remove devices after connect() has been called.")
        if 0 <= rank < len(self.devices):
            self.devices[rank] = None
        return self


    def connect(self, timeout: float = 60.0) -> "Cluster":
        if self.state == ClusterState.READY:
            logger.warning("Already connected.")
            return self

        if self.world_size == 1:
            self.state = ClusterState.READY
            logger.info("Single-machine mode — no network connection needed.")
            return self

        self.state = ClusterState.CONNECTING
        logger.info(
            f"[rank {self.rank}] Connecting... "
            f"world_size={self.world_size}, timeout={timeout}s"
        )

        try:
            from energizer.distributed.world import World
            self._world = World.init(
                rank       = self.rank,
                world_size = self.world_size,
                addr       = self.addr,
                port       = self.port,
                timeout    = timeout,
            )
        except Exception as e:
            self.state = ClusterState.IDLE
            raise RuntimeError(f"Cluster.connect() failed: {e}") from e

        for device in self.devices:
            if device is not None:
                device.status       = DeviceStatus.CONNECTED
                device.connected_at = time.time()

        self.state = ClusterState.READY
        logger.info(f"[rank {self.rank}] Cluster ready. {self.world_size} device(s) connected.")
        return self

    def close(self) -> None:
        if self._world is not None:
            try:
                self._world.close()
            except Exception:
                pass
            self._world = None

        for device in self.devices:
            if device is not None:
                device.status = DeviceStatus.UNKNOWN

        self.state = ClusterState.CLOSED
        logger.info(f"[rank {self.rank}] Cluster closed.")

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def ping_all(self, timeout: float = 2.0) -> dict[int, bool]:
        results = {}
        for device in self.devices:
            if device is None:
                continue
            if device.is_local:
                device.status     = DeviceStatus.CONNECTED
                device.latency_ms = 0.0
                results[device.rank] = True
            else:
                ok = device.ping(timeout=timeout)
                results[device.rank] = ok
                status = "✓" if ok else "✗"
                logger.info(
                    f"  {status} rank={device.rank} {device.addr}:{device.port} "
                    f"latency={device.latency_ms:.1f}ms"
                )
        return results

    def barrier(self) -> None:
        self._require_ready("barrier")
        if self.world_size == 1:
            return
        self._world.barrier()

    def fit(
        self,
        model,
        dataloader,
        epochs: int = 1,
        optimizer=None,
        loss_fn=None,
    ) -> None:
        """
        Distributed training. Each device processes a shard of every batch,
        gradients are averaged via AllReduce, and weights stay synchronised.

        For world_size=1, this is identical to single-machine training.
        For world_size>1, this wraps the model in DataParallel automatically.

        Args:
            model:      An energizer Module (already on the correct device).
            dataloader: Iterable of (x, y) batches.
            epochs:     Number of full passes over the data.
            optimizer:  An energizer Optimizer. If None, raises ValueError.
            loss_fn:    An energizer loss function. If None, raises ValueError.

        Example:
            cluster.fit(model, dataloader, epochs=10,
                        optimizer=Adam(model.parameters(), lr=1e-3),
                        loss_fn=MSELoss())
        """
        self._require_ready("fit")

        if optimizer is None:
            raise ValueError("cluster.fit() requires an optimizer.")
        if loss_fn is None:
            raise ValueError("cluster.fit() requires a loss_fn.")

        self.state = ClusterState.TRAINING

        try:
            if self.world_size == 1:
                self._fit_local(model, dataloader, epochs, optimizer, loss_fn)
            else:
                self._fit_distributed(model, dataloader, epochs, optimizer, loss_fn)
        finally:
            self.state = ClusterState.READY

    def _fit_local(self, model, dataloader, epochs, optimizer, loss_fn) -> None:
        from energizer import Tensor

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches  = 0
            model.train()

            for x, y in dataloader:
                if not isinstance(x, Tensor):
                    x = Tensor(x, requires_grad=False)
                if not isinstance(y, Tensor):
                    y = Tensor(y, requires_grad=False)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            avg = total_loss / max(n_batches, 1)
            logger.info(f"[rank 0] epoch {epoch}/{epochs}  loss={avg:.4f}")

    def _fit_distributed(self, model, dataloader, epochs, optimizer, loss_fn) -> None:
        try:
            from energizer.distributed.parallel import DataParallel
        except ImportError:
            raise ImportError(
                "energizer.distributed.parallel is required for multi-device training. "
                "Make sure Phase 2 (comm.py) and Phase 3 (parallel.py) are implemented."
            )

        parallel_model = DataParallel(model, world=self._world)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches  = 0
            model.train()

            for batch_idx, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                pred = parallel_model(x, batch_idx=batch_idx)
                loss = loss_fn(pred, y)
                loss.backward()
                parallel_model.sync_gradients()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            self._world.barrier()

            if self.rank == 0:
                avg = total_loss / max(n_batches, 1)
                logger.info(f"[rank 0] epoch {epoch}/{epochs}  loss={avg:.4f}")

    def predict(self, model, inputs, batch_size: Optional[int] = None):
        """
        Distributed inference. Splits inputs across devices, collects results
        back on rank 0.

        For world_size=1, runs locally with no overhead.

        Args:
            model:      An energizer Module (eval mode recommended).
            inputs:     A Tensor or list of Tensors.
            batch_size: Optional per-device batch size override.

        Returns:
            Collected predictions on rank 0 (None on worker ranks).

        Example:
            outputs = cluster.predict(model, x_test)
        """
        self._require_ready("predict")

        self.state = ClusterState.PREDICTING
        try:
            if self.world_size == 1:
                model.eval()
                return model(inputs)
            else:
                return self._predict_distributed(model, inputs, batch_size)
        finally:
            self.state = ClusterState.READY

    def _predict_distributed(self, model, inputs, batch_size):
        try:
            from energizer.distributed import comm
        except ImportError:
            raise ImportError(
                "energizer.distributed.comm is required for distributed inference."
            )

        import numpy as np

        model.eval()

        if self.rank == 0:
            arr = inputs.numpy() if hasattr(inputs, "numpy") else inputs
            shards = np.array_split(arr, self.world_size)
        else:
            shards = None

        local_shard = comm.scatter(shards, src=0, world=self._world)

        from energizer import Tensor
        shard_tensor = Tensor(local_shard, device=model.device
                              if hasattr(model, "device") else "cpu")
        local_output = model(shard_tensor)

        all_outputs = comm.gather(local_output.numpy(), dst=0, world=self._world)

        if self.rank == 0:
            from energizer import Tensor
            return Tensor(np.concatenate(all_outputs, axis=0))
        return None

    def broadcast_weights(self, model) -> None:
        self._require_ready("broadcast_weights")
        if self.world_size == 1:
            return

        try:
            from energizer.distributed import comm
        except ImportError:
            raise ImportError("energizer.distributed.comm is required.")

        import numpy as np

        for param in model.parameters():
            arr = param.data if not hasattr(param.data, "numpy") else param.data
            if hasattr(arr, "tolist"):
                import numpy as np
                arr = np.array(arr)
            synced = comm.broadcast(arr, src=0, world=self._world)
            param.data = synced

        logger.info(f"[rank {self.rank}] Weights broadcast complete.")

    def _require_ready(self, op: str) -> None:
        if self.state == ClusterState.CLOSED:
            raise RuntimeError(f"Cannot call {op}() on a closed cluster.")
        if self.state == ClusterState.CONNECTING:
            raise RuntimeError(f"Cannot call {op}() while cluster is connecting.")

    @property
    def world_size(self) -> int:
        return sum(1 for d in self.devices if d is not None)

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    @property
    def is_ready(self) -> bool:
        return self.state == ClusterState.READY

    @property
    def coordinator_device(self) -> Optional[DeviceInfo]:
        return self.devices[0] if self.devices else None

    def device(self, rank: int) -> Optional[DeviceInfo]:
        if 0 <= rank < len(self.devices):
            return self.devices[rank]
        return None

    def status(self) -> str:
        lines = [
            f"Cluster  state={self.state}  world_size={self.world_size}",
            "─" * 60,
        ]
        for device in self.devices:
            if device is not None:
                marker = "▶" if device.rank == self.rank else " "
                lines.append(f"{marker}{device.summary()}")
        lines.append("─" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        devices_repr = ", ".join(
            repr(d) for d in self.devices if d is not None
        )
        return (
            f"Cluster(rank={self.rank}, world_size={self.world_size}, "
            f"state={self.state}, devices=[{devices_repr}])"
        )