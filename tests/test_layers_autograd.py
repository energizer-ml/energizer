"""Forward + gradients of norm / pool / convT / embedding / attention layers
against PyTorch, on CPU and MLX."""

import unittest

import numpy as np
import torch
import torch.nn as nn

import energizer as ez
from energizer.layers.transformer import MultiheadAttention
from energizer._mlx import MLX_AVAILABLE

DEVICES = ("cpu", "mlx") if MLX_AVAILABLE else ("cpu",)
RNG = np.random.default_rng(0)


def arr(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def close(a, b, msg, atol=2e-4):
    a = a.numpy() if hasattr(a, "numpy") and not isinstance(a, np.ndarray) else a
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    np.testing.assert_allclose(np.asarray(a), b, rtol=2e-4, atol=atol, err_msg=msg)


def set_w(layer, name, value):
    getattr(layer, name).data = np.array(value.detach().numpy())


class TestLayerGradients(unittest.TestCase):
    def _compare(self, e_layer, t_layer, x_np, device, params=()):
        """Forward, input grad and listed (ez_name, torch_param) grads match."""
        e_layer.to(device)
        x_t = torch.tensor(x_np, requires_grad=True)
        x_e = ez.Tensor(x_np, device=device, requires_grad=True)
        y_t, y_e = t_layer(x_t), e_layer(x_e)
        close(y_e.numpy(), y_t, "forward")
        up = arr(*y_t.shape)
        (y_t * torch.tensor(up)).sum().backward()
        (y_e * ez.Tensor(up, device=device)).sum().backward()
        close(x_e.grad.numpy(), x_t.grad, "dx")
        for name, tp in params:
            obj = e_layer
            for part in name.split("."):
                obj = getattr(obj, part)
            close(obj.grad.numpy(), tp.grad, f"d{name}")

    def test_layernorm(self):
        for device in DEVICES:
            with self.subTest(device=device):
                t = nn.LayerNorm(8)
                t.weight.data = torch.tensor(arr(8))
                t.bias.data = torch.tensor(arr(8))
                e = ez.LayerNorm(8)
                set_w(e, "gamma", t.weight)
                set_w(e, "beta", t.bias)
                self._compare(
                    e,
                    t,
                    arr(4, 5, 8),
                    device,
                    [("gamma", t.weight), ("beta", t.bias)],
                )

    def test_batchnorm(self):
        for device in DEVICES:
            for cls_e, cls_t, shape in (
                (ez.BatchNorm1d, nn.BatchNorm1d, (6, 4)),
                (ez.BatchNorm1d, nn.BatchNorm1d, (6, 4, 7)),
                (ez.BatchNorm2d, nn.BatchNorm2d, (3, 4, 5, 5)),
            ):
                for training in (True, False):
                    with self.subTest(device=device, shape=shape, training=training):
                        t, e = cls_t(4), cls_e(4)
                        t.weight.data = torch.tensor(arr(4))
                        t.bias.data = torch.tensor(arr(4))
                        set_w(e, "gamma", t.weight)
                        set_w(e, "beta", t.bias)
                        if training:
                            t.train(), e.train()
                        else:
                            t.running_mean = torch.tensor(arr(4))
                            t.running_var = torch.tensor(np.abs(arr(4)) + 0.5)
                            e.running_mean = t.running_mean.numpy().copy()
                            e.running_var = t.running_var.numpy().copy()
                            t.eval(), e.eval()
                        self._compare(
                            e,
                            t,
                            arr(*shape),
                            device,
                            [("gamma", t.weight), ("beta", t.bias)],
                        )
                        if training:  # running stats follow PyTorch's update
                            close(np.asarray(e.running_mean), t.running_mean, "rmean")
                            close(np.asarray(e.running_var), t.running_var, "rvar")

    def test_pools(self):
        for device in DEVICES:
            for k, s, p in ((2, 2, 0), (3, 1, 1), (3, 2, 1)):
                with self.subTest(device=device, k=k, s=s, p=p):
                    self._compare(
                        ez.MaxPool2d(k, stride=s, padding=p),
                        nn.MaxPool2d(k, stride=s, padding=p),
                        arr(2, 3, 8, 7),
                        device,
                    )
                    self._compare(
                        ez.AvgPool2d(k, stride=s, padding=p),
                        nn.AvgPool2d(k, stride=s, padding=p),
                        arr(2, 3, 8, 7),
                        device,
                    )

    def test_conv_transpose(self):
        for device in DEVICES:
            for stride, padding, outpad in ((1, 0, 0), (2, 1, 0), (2, 1, 1), (3, 0, 2)):
                with self.subTest(device=device, s=stride, p=padding, op=outpad):
                    t = nn.ConvTranspose2d(
                        3, 4, 3, stride=stride, padding=padding, output_padding=outpad
                    )
                    e = ez.ConvTranspose2d(
                        3, 4, 3, stride=stride, padding=padding, output_padding=outpad
                    )
                    set_w(e, "weight", t.weight)
                    set_w(e, "bias", t.bias)
                    self._compare(
                        e,
                        t,
                        arr(2, 3, 5, 4),
                        device,
                        [("weight", t.weight), ("bias", t.bias)],
                    )

    def test_embedding(self):
        for device in DEVICES:
            with self.subTest(device=device):
                t = nn.Embedding(10, 6, padding_idx=0)
                e = ez.Embedding(10, 6, padding_idx=0)
                set_w(e, "weight", t.weight)
                e.to(device)
                idx = np.array([[1, 2, 2, 0], [9, 1, 0, 3]])
                y_t = t(torch.tensor(idx))
                y_e = e(ez.Tensor(idx.astype(np.float32), device=device))
                close(y_e.numpy(), y_t, "forward")
                up = arr(*y_t.shape)
                (y_t * torch.tensor(up)).sum().backward()
                (y_e * ez.Tensor(up, device=device)).sum().backward()
                close(e.weight.grad.numpy(), t.weight.grad, "dweight")

    def _mha_pair(self, d=8, h=2):
        t = nn.MultiheadAttention(d, h, dropout=0.0, batch_first=True)
        e = MultiheadAttention(d, h, dropout=0.0)
        w = t.in_proj_weight.detach().numpy()
        b = t.in_proj_bias.detach().numpy()
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            getattr(e, name).weight.data = w[i * d : (i + 1) * d].copy()
            getattr(e, name).bias.data = b[i * d : (i + 1) * d].copy()
        e.out_proj.weight.data = t.out_proj.weight.detach().numpy().copy()
        e.out_proj.bias.data = t.out_proj.bias.detach().numpy().copy()
        return e, t

    def test_multihead_attention(self):
        for device in DEVICES:
            with self.subTest(device=device):
                e, t = self._mha_pair()
                e.to(device)
                x_np = arr(2, 5, 8)
                x_t = torch.tensor(x_np, requires_grad=True)
                x_e = ez.Tensor(x_np, device=device, requires_grad=True)
                y_t, _ = t(x_t, x_t, x_t, need_weights=False)
                y_e = e(x_e, x_e, x_e)
                close(y_e.numpy(), y_t, "forward")
                y_t.sum().backward()
                y_e.sum().backward()
                close(x_e.grad.numpy(), x_t.grad, "dx", atol=1e-3)
                close(
                    e.out_proj.weight.grad.numpy(),
                    t.out_proj.weight.grad,
                    "d out_w",
                    atol=1e-3,
                )

    def test_dropout_backward_uses_mask(self):
        for device in DEVICES:
            with self.subTest(device=device):
                drop = ez.Dropout(0.5)
                drop.train()
                x = ez.Tensor(
                    np.ones((64, 64), np.float32), device=device, requires_grad=True
                )
                y = drop(x)
                y.sum().backward()
                # gradient equals the (scaled) mask that produced the output
                np.testing.assert_allclose(x.grad.numpy(), y.numpy())


if __name__ == "__main__":
    unittest.main()
