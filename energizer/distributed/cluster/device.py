"""
energizer/distributed/cluster/device.py
────────────────────────────────────────
DeviceInfo — represents one machine in the cluster.

Stores connection metadata, hardware info, and live health state.
Does not own any socket — that belongs to World/PeerInfo.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceRole(str, Enum):
    COORDINATOR = "coordinator"
    WORKER      = "worker"


class DeviceStatus(str, Enum):
    UNKNOWN     = "unknown"
    REACHABLE   = "reachable"
    UNREACHABLE = "unreachable"
    CONNECTED   = "connected"
    BUSY        = "busy"

def _probe_chip() -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def _probe_memory_gb() -> float:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        return int(out) / (1024 ** 3)
    except Exception:
        return 0.0


def _probe_hostname() -> str:
    try:
        return platform.node()
    except Exception:
        return "unknown"



@dataclass
class DeviceInfo:
    """
    Metadata for one machine in the Energizer cluster.

    Fields set at registration time:
        rank        Unique integer ID. 0 = coordinator.
        addr        IP address or hostname.
        port        Rendezvous port (default 29500).
        role        COORDINATOR or WORKER.

    Fields filled in after connection / probe:
        hostname    Human-readable machine name.
        chip        CPU/chip identifier, e.g. "Apple M3 Pro".
        memory_gb   Total RAM in gigabytes.
        status      Current reachability state.
        latency_ms  Last measured round-trip ping latency.
        connected_at Unix timestamp when connection was established.
    """

    rank:         int
    addr:         str
    port:         int         = 29500
    role:         DeviceRole  = DeviceRole.WORKER

    # Filled after probe
    hostname:     str         = field(default="unknown", compare=False)
    chip:         str         = field(default="unknown", compare=False)
    memory_gb:    float       = field(default=0.0,       compare=False)

    # Live state
    status:       DeviceStatus = field(default=DeviceStatus.UNKNOWN, compare=False)
    latency_ms:   float        = field(default=-1.0,    compare=False)
    connected_at: float        = field(default=0.0,     compare=False)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def local(cls, rank: int, port: int = 29500) -> "DeviceInfo":
        role = DeviceRole.COORDINATOR if rank == 0 else DeviceRole.WORKER
        d = cls(
            rank      = rank,
            addr      = "127.0.0.1",
            port      = port,
            role      = role,
            hostname  = _probe_hostname(),
            chip      = _probe_chip(),
            memory_gb = _probe_memory_gb(),
            status    = DeviceStatus.CONNECTED,
        )
        d.connected_at = time.time()
        return d

    @classmethod
    def remote(
        cls,
        rank: int,
        addr: str,
        port: int = 29500,
        hostname: Optional[str] = None,
    ) -> "DeviceInfo":
        role = DeviceRole.COORDINATOR if rank == 0 else DeviceRole.WORKER
        return cls(
            rank     = rank,
            addr     = addr,
            port     = port,
            role     = role,
            hostname = hostname or addr,
            status   = DeviceStatus.UNKNOWN,
        )

    # ── Health ────────────────────────────────────────────────────────────────

    def ping(self, timeout: float = 2.0) -> bool:
        import socket
        t0 = time.perf_counter()
        try:
            with socket.create_connection((self.addr, self.port), timeout=timeout):
                pass
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            self.status     = DeviceStatus.REACHABLE
            return True
        except OSError:
            self.latency_ms = -1.0
            self.status     = DeviceStatus.UNREACHABLE
            return False

    def icmp_ping(self, timeout: float = 2.0) -> bool:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", str(int(timeout * 1000)), self.addr],
                capture_output=True,
                timeout=timeout + 1,
            )
            if result.returncode == 0:
                # Parse round-trip time from output
                out = result.stdout.decode()
                for part in out.split():
                    if part.startswith("time="):
                        self.latency_ms = float(part.split("=")[1])
                        break
                self.status = DeviceStatus.REACHABLE
                return True
        except Exception:
            pass
        self.status = DeviceStatus.UNREACHABLE
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    @property
    def is_local(self) -> bool:
        return self.addr in ("127.0.0.1", "localhost", "0.0.0.0")

    @property
    def is_connected(self) -> bool:
        return self.status == DeviceStatus.CONNECTED

    @property
    def uptime_s(self) -> float:
        if self.connected_at == 0.0:
            return 0.0
        return time.time() - self.connected_at

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lat = f"{self.latency_ms:.1f}ms" if self.latency_ms >= 0 else "n/a"
        mem = f"{self.memory_gb:.0f}GB"  if self.memory_gb  > 0  else "?"
        return (
            f"  rank={self.rank}  {self.role.value:<12}  "
            f"{self.addr}:{self.port}  "
            f"hostname={self.hostname}  "
            f"chip={self.chip}  "
            f"mem={mem}  "
            f"status={self.status.value}  "
            f"latency={lat}"
        )

    def __repr__(self) -> str:
        return (
            f"DeviceInfo(rank={self.rank}, addr='{self.addr}:{self.port}', "
            f"role={self.role.value}, status={self.status.value})"
        )