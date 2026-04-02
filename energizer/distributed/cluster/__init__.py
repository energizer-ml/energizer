"""
energizer/distributed/cluster/__init__.py
──────────────────────────────────────────
Public API for the Energizer cluster manager.

Quick start:

    from energizer.distributed.cluster import Cluster

    # MacBook Pro (rank 0):
    cluster = Cluster.coordinator(port=29500)
    cluster.add_device(rank=1, addr="192.168.1.20", hostname="mac-mini")
    cluster.connect()
    print(cluster.status())

    # Mac Mini (rank 1):
    cluster = Cluster.worker(rank=1, coordinator_addr="192.168.1.5")
    cluster.connect()
"""

from .device  import DeviceInfo, DeviceRole, DeviceStatus
from .manager import Cluster, ClusterState

__all__ = [
    "Cluster",
    "ClusterState",
    "DeviceInfo",
    "DeviceRole",
    "DeviceStatus",
]