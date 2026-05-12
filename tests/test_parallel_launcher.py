import os
import sys
import unittest

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from energizer.parallel.launcher import (
    ParallelLaunchConfig,
    ParallelMonitor,
    build_mlx_launch_command,
)


class TestParallelLauncher(unittest.TestCase):
    def test_launcher_uses_bash_wrapper_to_avoid_local_path_rewrite(self):
        config = ParallelLaunchConfig(
            hosts=["10.0.0.1", "10.0.0.2"],
            cwd="/Users/Shared/projets/energizer",
            python=".venv/bin/python",
            script="examples/train.py",
            script_args=("--epochs", "1"),
        )

        cmd = build_mlx_launch_command(config)

        self.assertEqual(
            cmd[:7],
            [
                "mlx.launch",
                "--backend",
                "ring",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--cwd",
                "/Users/Shared/projets/energizer",
            ],
        )
        self.assertIn("--no-verify-script", cmd)
        self.assertEqual(cmd[-3:-1], ["bash", "-lc"])
        self.assertIn(".venv/bin/python", cmd[-1])
        self.assertIn("examples/train.py", cmd[-1])
        self.assertIn("--epochs", cmd[-1])

    def test_parallel_monitor_parses_rank_lines_and_failures(self):
        monitor = ParallelMonitor(["10.0.0.1", "10.0.0.2"])

        monitor.record("rank=1 size=2 local=2.0 all_sum=3.0")
        monitor.record("[WARN] Node with rank 0 exited with code 1")

        self.assertEqual(monitor.world_size, 2)
        self.assertEqual(monitor.ranks[1].status, "live")
        self.assertEqual(monitor.ranks[0].status, "failed")
        self.assertEqual(monitor.ranks[0].exit_code, 1)


if __name__ == "__main__":
    unittest.main()
