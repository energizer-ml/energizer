import queue
import threading
import time
import unittest

import numpy as np

import energizer


def _start_monitor(port: int):
    server = energizer.TelemetryServer(addr="127.0.0.1", port=port)
    server.start()
    return server


def _build_world_pair(port: int, monitor_port: int):
    worlds = queue.Queue()
    errors = queue.Queue()

    def coordinator():
        try:
            worlds.put(
                energizer.World.init(
                    0,
                    2,
                    "127.0.0.1",
                    port,
                    monitor_addr="127.0.0.1",
                    monitor_port=monitor_port,
                    node_name="macbook",
                )
            )
        except Exception as exc:
            errors.put(exc)

    thread = threading.Thread(target=coordinator, daemon=True)
    thread.start()
    worker = energizer.World.init(
        1,
        2,
        "127.0.0.1",
        port,
        monitor_addr="127.0.0.1",
        monitor_port=monitor_port,
        node_name="macmini",
    )
    coordinator_world = worlds.get(timeout=5)
    thread.join(timeout=5)

    if not errors.empty():
        raise errors.get()

    return coordinator_world, worker


class TestDistributedMonitor(unittest.TestCase):
    def test_monitor_receives_rank_traffic(self):
        monitor = _start_monitor(29660)
        world0, world1 = _build_world_pair(29661, 29660)
        try:
            errors = queue.Queue()

            def run(world, value):
                try:
                    world.allreduce(value)
                except Exception as exc:
                    errors.put(exc)

            threads = [
                threading.Thread(
                    target=run, args=(world0, np.array([1.0, 2.0], dtype=np.float32))
                ),
                threading.Thread(
                    target=run, args=(world1, np.array([3.0, 4.0], dtype=np.float32))
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            if not errors.empty():
                raise errors.get()

            deadline = time.time() + 2.0
            snapshot = {"ranks": []}
            while time.time() < deadline:
                snapshot = monitor.metrics.snapshot()
                if len(snapshot["ranks"]) == 2 and all(
                    rank["allreduce_calls"] >= 1 for rank in snapshot["ranks"]
                ):
                    break
                time.sleep(0.05)

            self.assertEqual(len(snapshot["ranks"]), 2)
            by_rank = {rank["rank"]: rank for rank in snapshot["ranks"]}
            self.assertEqual(by_rank[0]["node"], "macbook")
            self.assertEqual(by_rank[1]["node"], "macmini")
            self.assertGreaterEqual(by_rank[0]["bytes_received"], 8)
            self.assertGreaterEqual(by_rank[1]["bytes_sent"], 8)
            self.assertGreaterEqual(by_rank[0]["allreduce_calls"], 1)
            self.assertGreaterEqual(by_rank[1]["allreduce_calls"], 1)
        finally:
            world0.close()
            world1.close()
            monitor.stop()


if __name__ == "__main__":
    unittest.main()
