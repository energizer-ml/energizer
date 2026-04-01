from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque


class TelemetryClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        rank: int,
        world_size: int,
        node_name: str | None = None,
        timeout: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.node_name = node_name or socket.gethostname()
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self._sock = sock

    def emit(self, event: str, data: dict | None = None) -> None:
        payload = {
            "ts": time.time(),
            "rank": self.rank,
            "world_size": self.world_size,
            "node": self.node_name,
            "event": event,
            "data": data or {},
        }
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            if self._sock is None:
                try:
                    self.connect()
                except OSError:
                    return
            try:
                assert self._sock is not None
                self._sock.sendall(line)
            except OSError:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def close(self) -> None:
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class ClusterMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self._ranks: dict[int, dict] = {}

    def ingest(self, event: dict) -> None:
        rank = int(event["rank"])
        now = float(event["ts"])
        data = event.get("data", {})
        with self._lock:
            state = self._ranks.setdefault(
                rank,
                {
                    "rank": rank,
                    "node": event.get("node", ""),
                    "world_size": int(event.get("world_size", 1)),
                    "last_event": "",
                    "last_ts": now,
                    "events": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "allreduce_calls": 0,
                    "broadcast_calls": 0,
                    "allreduce_ms_total": 0.0,
                    "history": deque(maxlen=20),
                },
            )
            state["node"] = event.get("node", state["node"])
            state["world_size"] = int(event.get("world_size", state["world_size"]))
            state["last_event"] = event["event"]
            state["last_ts"] = now
            state["events"] += 1
            state["bytes_sent"] += int(data.get("bytes_sent", 0))
            state["bytes_received"] += int(data.get("bytes_received", 0))
            if event["event"] == "allreduce":
                state["allreduce_calls"] += 1
                state["allreduce_ms_total"] += float(data.get("duration_ms", 0.0))
            if event["event"] == "broadcast":
                state["broadcast_calls"] += 1
            state["history"].append(
                {"event": event["event"], "ts": now, "data": data}
            )

    def snapshot(self) -> dict:
        with self._lock:
            ranks = []
            for rank in sorted(self._ranks):
                state = self._ranks[rank]
                ranks.append(
                    {
                        "rank": state["rank"],
                        "node": state["node"],
                        "world_size": state["world_size"],
                        "last_event": state["last_event"],
                        "last_ts": state["last_ts"],
                        "events": state["events"],
                        "bytes_sent": state["bytes_sent"],
                        "bytes_received": state["bytes_received"],
                        "allreduce_calls": state["allreduce_calls"],
                        "broadcast_calls": state["broadcast_calls"],
                        "avg_allreduce_ms": (
                            state["allreduce_ms_total"] / state["allreduce_calls"]
                            if state["allreduce_calls"]
                            else 0.0
                        ),
                        "history": list(state["history"]),
                    }
                )
        return {"started_at": self.started_at, "ranks": ranks}


class TelemetryServer:
    def __init__(self, addr: str = "0.0.0.0", port: int = 29650, *, timeout: float = 1.0):
        self.addr = addr
        self.port = port
        self.timeout = timeout
        self.metrics = ClusterMetrics()
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._client_threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.addr, self.port))
        listener.listen()
        listener.settimeout(self.timeout)
        self._listener = listener
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(target=self._client_loop, args=(conn,), daemon=True)
            self._client_threads.append(thread)
            thread.start()

    def _client_loop(self, conn: socket.socket) -> None:
        conn.settimeout(self.timeout)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if not raw:
                        continue
                    self.metrics.ingest(json.loads(raw.decode("utf-8")))
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None
        for thread in self._client_threads:
            thread.join(timeout=1)
        self._client_threads.clear()
