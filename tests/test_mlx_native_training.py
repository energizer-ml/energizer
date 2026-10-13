"""MLX-native loss / optimizer paths must match the NumPy reference."""

import unittest

import numpy as np

import energizer as ez
from energizer.neural_network import Parameter
from energizer._mlx import MLX_AVAILABLE


@unittest.skipUnless(MLX_AVAILABLE, "MLX not available")
class TestMLXNativeTraining(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.logits = rng.standard_normal((8, 5)).astype(np.float32)
        self.labels = rng.integers(0, 5, 8).astype(np.float32)
        self.onehot = np.eye(5, dtype=np.float32)[self.labels.astype(int)]

    def _ce(self, device, target, reduction):
        x = ez.Tensor(self.logits, device=device, requires_grad=True)
        loss = ez.CrossEntropyLoss(reduction=reduction)(
            x, ez.Tensor(target, device=device)
        )
        loss.sum().backward() if loss.data.size > 1 else loss.backward()
        return loss.numpy(), x.grad.numpy()

    def test_cross_entropy_matches_cpu(self):
        for reduction in ("mean", "sum", "none"):
            for target in (self.labels, self.onehot):
                with self.subTest(reduction=reduction, target_ndim=target.ndim):
                    l_cpu, g_cpu = self._ce("cpu", target, reduction)
                    l_mlx, g_mlx = self._ce("mlx", target, reduction)
                    np.testing.assert_allclose(l_mlx, l_cpu, rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(g_mlx, g_cpu, rtol=1e-5, atol=1e-5)

    def test_adam_step_matches_cpu_and_stays_on_mlx(self):
        import mlx.core as mx

        results = {}
        for device in ("cpu", "mlx"):
            w = Parameter(
                np.ones((3, 3), dtype=np.float32), requires_grad=True, device=device
            )
            x = ez.Tensor(np.arange(9, dtype=np.float32).reshape(3, 3), device=device)
            opt = ez.Adam([w], lr=0.1)
            for _ in range(3):
                opt.zero_grad()
                (w * x).sum().backward()
                opt.step()
            results[device] = w
        np.testing.assert_allclose(
            np.array(results["mlx"].data), np.asarray(results["cpu"].data), atol=1e-6
        )
        self.assertIsInstance(results["mlx"].data, mx.array)


class TestOptimizerParamGroups(unittest.TestCase):
    """Regression: SGD(list_of_params) used to iterate tensor rows (no-op / hang)."""

    def _devices(self):
        return ("cpu", "mlx") if MLX_AVAILABLE else ("cpu",)

    def test_sgd_updates_every_parameter(self):
        for device in self._devices():
            with self.subTest(device=device):
                ws = [
                    Parameter(
                        np.ones((2, 3), dtype=np.float32),
                        requires_grad=True,
                        device=device,
                    )
                    for _ in range(2)
                ]
                opt = ez.SGD(ws, lr=0.5)
                loss = (ws[0] * 2.0).sum() + (ws[1] * 3.0).sum()
                loss.backward()
                opt.step()
                np.testing.assert_allclose(np.array(ws[0].data), 1.0 - 0.5 * 2.0)
                np.testing.assert_allclose(np.array(ws[1].data), 1.0 - 0.5 * 3.0)
                self.assertEqual(len(opt.param_groups), 1)


if __name__ == "__main__":
    unittest.main()
