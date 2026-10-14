"""Conv1d / Conv2d forward + backward against PyTorch, on CPU and MLX."""

import unittest

import numpy as np
import torch
import torch.nn as nn

import energizer as ez
from energizer._mlx import MLX_AVAILABLE

DEVICES = ("cpu", "mlx") if MLX_AVAILABLE else ("cpu",)


def _check(test, ours, theirs, name, atol=1e-4):
    np.testing.assert_allclose(
        np.asarray(ours), theirs.detach().numpy(), rtol=1e-4, atol=atol, err_msg=name
    )


class TestConvAutograd(unittest.TestCase):
    def _run(self, dim, device, stride, padding, bias):
        rng = np.random.default_rng(0)
        cin, cout, k = 3, 5, 3
        shape = (2, cin) + ((9,) if dim == 1 else (8, 7))
        x_np = rng.standard_normal(shape).astype(np.float32)

        t_layer = (nn.Conv1d if dim == 1 else nn.Conv2d)(
            cin, cout, k, stride=stride, padding=padding, bias=bias
        )
        e_layer = (ez.Conv1d if dim == 1 else ez.Conv2d)(
            cin, cout, k, stride=stride, padding=padding, bias=bias
        )
        e_layer.weight.data = np.array(t_layer.weight.detach().numpy())
        if bias:
            e_layer.bias.data = np.array(t_layer.bias.detach().numpy())
        e_layer.to(device)

        x_t = torch.tensor(x_np, requires_grad=True)
        x_e = ez.Tensor(x_np, device=device, requires_grad=True)
        y_t, y_e = t_layer(x_t), e_layer(x_e)
        _check(self, y_e.numpy(), y_t, "forward")

        # a non-trivial upstream gradient
        w_np = rng.standard_normal(tuple(y_t.shape)).astype(np.float32)
        (y_t * torch.tensor(w_np)).sum().backward()
        (y_e * ez.Tensor(w_np, device=device)).sum().backward()
        _check(self, x_e.grad.numpy(), x_t.grad, "dx")
        _check(self, e_layer.weight.grad.numpy(), t_layer.weight.grad, "dw")
        if bias:
            _check(self, e_layer.bias.grad.numpy(), t_layer.bias.grad, "db")

    def test_all(self):
        for dim in (1, 2):
            for device in DEVICES:
                for stride, padding in ((1, 0), (1, 1), (2, 1)):
                    for bias in (True, False):
                        with self.subTest(
                            dim=dim,
                            device=device,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ):
                            self._run(dim, device, stride, padding, bias)


if __name__ == "__main__":
    unittest.main()
