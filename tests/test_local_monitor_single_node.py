import time
import unittest

import energizer


class TestLocalMonitorSingleNode(unittest.TestCase):
    def test_single_machine_can_report_to_monitor(self):
        monitor = energizer.TelemetryServer(addr="127.0.0.1", port=29670)
        monitor.start()
        world = None
        try:
            world = energizer.World.init(
                rank=0,
                world_size=1,
                addr="127.0.0.1",
                port=29671,
                monitor_addr="127.0.0.1",
                monitor_port=29670,
                node_name="local-mac",
            )

            model = energizer.DataParallel(energizer.Linear(4, 2), world)
            world._emit("single_node_smoke_test", {"source": "local"})

            deadline = time.time() + 2.0
            snapshot = {"ranks": []}
            while time.time() < deadline:
                snapshot = monitor.metrics.snapshot()
                if snapshot["ranks"]:
                    history = snapshot["ranks"][0]["history"]
                    events = {item["event"] for item in history}
                    if {
                        "world_init",
                        "model_state_sync",
                        "single_node_smoke_test",
                    }.issubset(events):
                        break
                time.sleep(0.05)

            self.assertEqual(len(snapshot["ranks"]), 1)
            rank_state = snapshot["ranks"][0]
            self.assertEqual(rank_state["rank"], 0)
            self.assertEqual(rank_state["node"], "local-mac")

            history = rank_state["history"]
            events = {item["event"] for item in history}
            self.assertIn("world_init", events)
            self.assertIn("model_state_sync", events)
            self.assertIn("single_node_smoke_test", events)
        finally:
            if world is not None:
                world.close()
            monitor.stop()
