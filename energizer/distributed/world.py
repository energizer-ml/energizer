from __future__ import annotations

import io
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from energizer.backend import backend


_HEADER = struct.Struct("!Q")


def _send_message(sock: socket.socket, payload: Any) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.array(payload, dtype=object), allow_pickle=True)
    data = buffer.getvalue()
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving distributed payload")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_message(sock: socket.socket) -> Any:
    header = _recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    payload = _recv_exact(sock, size)
    buffer = io.BytesIO(payload)
    array = np.load(buffer, allow_pickle=True)
    return array.item()


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    device = "gpu" if "mlx" in str(type(value)) else "cpu"
    return backend.to_numpy(value, device).astype(np.float32, copy=False)


def _restore_backend(array: np.ndarray, template):
    if isinstance(template, np.ndarray):
        return array.astype(template.dtype, copy=False)
    device = "gpu" if "mlx" in str(type(template)) else "cpu"
    restored = backend.array(array, device)
    if isinstance(template, np.ndarray):
        return restored.astype(template.dtype, copy=False)
    return restored


class World:
    def __init__(
        self,
        rank: int,
        world_size: int,
        addr: str,
        port: int,
        sockets: dict[int, socket.socket],
        listener: socket.socket | None = None,
    ):
        self.rank = rank
        self.size = world_size
        self.world_size = world_size
        self.addr = addr
        self.port = port
        self.coordinator_rank = 0
        self._sockets = sockets
        self._listener = listener
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def is_coordinator(self) -> bool:
        return self.rank == self.coordinator_rank

    @classmethod
    def init(
        cls,
        rank: int,
        world_size: int,
        addr: str,
        port: int,
        *,
        timeout: float = 30.0,
        retry_interval: float = 0.2,
        max_retries: int = 50,
        monitor_addr: str | None = None,
        monitor_port: int | None = None,
        node_name: str | None = None,
    ) -> "World":
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")

        if world_size == 1:
            world = cls(rank, world_size, addr, port, sockets={})
            world._emit(
                "world_init",
                {"addr": addr, "port": port, "coordinator_rank": world.coordinator_rank},
            )
            return world

        if rank == 0:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((addr, port))
            listener.listen(world_size - 1)
            listener.settimeout(timeout)

            sockets: dict[int, socket.socket] = {}
            try:
                while len(sockets) < world_size - 1:
                    conn, _ = listener.accept()
                    conn.settimeout(timeout)
                    hello = _recv_message(conn)
                    peer_rank = hello["rank"]
                    if hello["world_size"] != world_size:
                        conn.close()
                        raise ValueError("Worker world_size does not match coordinator")
                    if peer_rank in sockets:
                        conn.close()
                        raise ValueError(f"Duplicate rank connection received: {peer_rank}")
                    sockets[peer_rank] = conn
                    _send_message(conn, {"ok": True})
            except Exception:
                for conn in sockets.values():
                    conn.close()
                listener.close()
                raise
            world = cls(
                rank,
                world_size,
                addr,
                port,
                sockets=sockets,
                listener=listener,
            )
            world._emit(
                "world_init",
                {
                    "addr": addr,
                    "port": port,
                    "coordinator_rank": world.coordinator_rank,
                    "connected_ranks": sorted(sockets),
                },
            )
            return world

        last_error = None
        for _ in range(max_retries):
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(timeout)
                conn.connect((addr, port))
                _send_message(conn, {"rank": rank, "world_size": world_size})
                _recv_message(conn)
                world = cls(
                    rank,
                    world_size,
                    addr,
                    port,
                    sockets={0: conn},
                )
                world._emit(
                    "world_init",
                    {"addr": addr, "port": port, "coordinator_rank": world.coordinator_rank},
                )
                return world
            except OSError as exc:
                last_error = exc
                try:
                    conn.close()
                except Exception:
                    pass
                threading.Event().wait(retry_interval)
        raise ConnectionError(
            f"Rank {rank} failed to connect to coordinator at {addr}:{port}"
        ) from last_error

    def _next_tag(self) -> int:
        tag = self._sequence
        self._sequence += 1
        return tag

    def report(self, event: str, data: dict | None = None) -> None:
        pass

    def allreduce(self, value):
        if self.world_size == 1:
            return value

        with self._lock:
            tag = self._next_tag()
            local = _as_numpy(value)
            local_bytes = int(local.nbytes)
            started = time.perf_counter()

            if self.is_coordinator:
                total = local.copy()
                bytes_received = 0
                for peer_rank in range(1, self.world_size):
                    message = _recv_message(self._sockets[peer_rank])
                    if message["op"] != "allreduce" or message["tag"] != tag:
                        raise RuntimeError("Received out-of-order allreduce payload")
                    peer_value = np.asarray(message["value"], dtype=np.float32)
                    bytes_received += int(peer_value.nbytes)
                    total += peer_value

                averaged = total / float(self.world_size)
                response = {"op": "allreduce_result", "tag": tag, "value": averaged}
                for peer_rank in range(1, self.world_size):
                    _send_message(self._sockets[peer_rank], response)
                self._emit(
                    "allreduce",
                    {
                        "tag": tag,
                        "bytes_sent": int(averaged.nbytes) * (self.world_size - 1),
                        "bytes_received": bytes_received,
                        "local_tensor_bytes": local_bytes,
                        "duration_ms": (time.perf_counter() - started) * 1000.0,
                        "participants": self.world_size,
                    },
                )
                return _restore_backend(averaged, value)

            coordinator = self._sockets[self.coordinator_rank]
            _send_message(
                coordinator,
                {"op": "allreduce", "tag": tag, "value": local},
            )
            response = _recv_message(coordinator)
            if response["op"] != "allreduce_result" or response["tag"] != tag:
                raise RuntimeError("Received invalid allreduce response from coordinator")
            result_value = np.asarray(response["value"], dtype=np.float32)
            self._emit(
                "allreduce",
                {
                    "tag": tag,
                    "bytes_sent": local_bytes,
                    "bytes_received": int(result_value.nbytes),
                    "local_tensor_bytes": local_bytes,
                    "duration_ms": (time.perf_counter() - started) * 1000.0,
                    "participants": self.world_size,
                },
            )
            return _restore_backend(response["value"], value)

    def broadcast_object(self, value, src: int = 0):
        if src != self.coordinator_rank:
            raise NotImplementedError("Only rank 0 broadcast is supported")
        if self.world_size == 1:
            return value

        with self._lock:
            tag = self._next_tag()
            if self.rank == src:
                payload = {"op": "broadcast", "tag": tag, "value": value}
                payload_bytes = len(repr(value).encode("utf-8"))
                for peer_rank in range(1, self.world_size):
                    _send_message(self._sockets[peer_rank], payload)
                self._emit(
                    "broadcast",
                    {
                        "tag": tag,
                        "bytes_sent": payload_bytes * (self.world_size - 1),
                        "bytes_received": 0,
                    },
                )
                return value

            response = _recv_message(self._sockets[src])
            if response["op"] != "broadcast" or response["tag"] != tag:
                raise RuntimeError("Received invalid broadcast payload")
            self._emit(
                "broadcast",
                {
                    "tag": tag,
                    "bytes_sent": 0,
                    "bytes_received": len(repr(response["value"]).encode("utf-8")),
                },
            )
            return response["value"]

    def close(self):
        self._emit("world_close", {})
        for conn in self._sockets.values():
            try:
                conn.close()
            except OSError:
                pass
        self._sockets.clear()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        
