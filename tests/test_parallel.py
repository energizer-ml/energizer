import os
import sys
import unittest

import numpy as np

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import energizer
from energizer.parallel import DeviceGroup


class TestParallel(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.rtol = 1e-5
        self.atol = 1e-6

    def assert_close(self, actual, expected):
        if isinstance(actual, energizer.Tensor):
            actual = actual.data
        if isinstance(expected, energizer.Tensor):
            expected = expected.data
        np.testing.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    def test_shard_and_gather_preserves_values(self):
        x = energizer.tensor(np.arange(24, dtype=np.float32).reshape(3, 8))

        sharded = energizer.shard(x, parts=3, axis=1)
        gathered = energizer.gather(sharded)

        self.assertEqual(sharded.shape, x.shape)
        self.assertEqual([s.shape[1] for s in sharded], [3, 3, 2])
        self.assert_close(gathered, x)

    def test_gather_backward_routes_gradients_to_source_tensor(self):
        x = energizer.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            requires_grad=True,
        )

        gathered = energizer.gather(energizer.shard(x, parts=2, axis=1))
        loss = gathered.sum()
        loss.backward()

        self.assert_close(x.grad, np.ones_like(x.data))

    def test_all_to_sharded_linear_matches_linear_after_gather(self):
        group = DeviceGroup.local(size=3)
        linear = energizer.Linear(5, 8, bias=True)
        parallel = energizer.shard_linear(linear, "all-to-sharded", group=group)
        x = energizer.tensor(np.random.randn(4, 5).astype(np.float32))

        expected = linear(x)
        actual = energizer.gather(parallel(x))

        self.assert_close(actual, expected)

    def test_sharded_to_all_linear_matches_linear(self):
        group = DeviceGroup.local(size=2)
        linear = energizer.Linear(6, 4, bias=True)
        parallel = energizer.shard_linear(linear, "sharded-to-all", group=group)
        x = energizer.tensor(np.random.randn(3, 6).astype(np.float32))

        x_sharded = energizer.shard(x, parts=2, axis=1)
        expected = linear(x)
        actual = parallel(x_sharded)

        self.assert_close(actual, expected)

    def test_tensor_parallel_pair_matches_two_linears(self):
        group = DeviceGroup.local(size=2)
        l1 = energizer.Linear(4, 6, bias=False)
        l2 = energizer.Linear(6, 3, bias=True)
        p1 = energizer.shard_linear(l1, "all-to-sharded", group=group)
        p2 = energizer.shard_linear(l2, "sharded-to-all", group=group)
        x = energizer.tensor(np.random.randn(5, 4).astype(np.float32))

        expected = l2(l1(x))
        actual = p2(p1(x))

        self.assert_close(actual, expected)

    def test_split_batch_shards_axis_zero(self):
        x = energizer.tensor(np.arange(20, dtype=np.float32).reshape(5, 4))

        sharded = energizer.split_batch(x, parts=2)

        self.assertEqual([shard.shape[0] for shard in sharded], [3, 2])
        self.assert_close(energizer.gather(sharded), x)


if __name__ == "__main__":
    unittest.main()
