#!/usr/bin/env python
"""Ablations of the MLX-native training path + model-size sweep.

Each MLX-native component is switched off in turn by re-introducing the host
round trip it removed, so the cost of each optimization is isolated:

  full             everything native (default)
  no_native_loss   cross-entropy computed in NumPy on host
  no_native_optim  gradients copied to host before the optimizer step
  none             both host paths (the pre-step-5 behaviour)
  cpu              NumPy backend, for reference

Sweeps MLP width to show where the GPU path starts to win.

    python experiments/run_ablation.py --widths 256 512 1024 2048 4096
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np

import energizer as ez
import energizer.layers.loss as loss_mod
from ezbench import timing
from ezbench.results import ResultWriter


@contextlib.contextmanager
def host_loss(enabled: bool):
    if not enabled:
        yield
        return
    orig = loss_mod._is_mlx
    loss_mod._is_mlx = lambda a: False  # forces the NumPy implementation
    try:
        yield
    finally:
        loss_mod._is_mlx = orig


def build(width, device):
    model = ez.Sequential(
        ez.Linear(width, width),
        ez.ReLU(),
        ez.Linear(width, width),
        ez.ReLU(),
        ez.Linear(width, 10),
    ).to(device)
    return model


def measure(width, batch, device, no_loss, no_optim, iters):
    import mlx.core as mx

    model = build(width, device)
    params = model.parameters()
    opt = ez.Adam(params, lr=1e-3)
    lf = ez.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    x = ez.Tensor(rng.standard_normal((batch, width), dtype=np.float32), device=device)
    y = ez.Tensor(rng.integers(0, 10, batch).astype(np.float32), device=device)

    def step():
        opt.zero_grad()
        with host_loss(no_loss):
            loss = lf(model(x), y)
        loss.backward()
        if no_optim and device == "mlx":
            for p in params:
                if p.grad is not None:
                    p.grad = ez.Tensor(np.array(p.grad.data), device="cpu")
        opt.step()
        if device == "mlx":
            mx.eval(loss.data, *[p.data for p in params])

    return timing.repeat(lambda: step, repeats=3, warmup=3, iters=iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--widths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096]
    )
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    out = ResultWriter("ablation")
    print("writing", out.path)
    variants = [
        ("full", "mlx", False, False),
        ("no_native_loss", "mlx", True, False),
        ("no_native_optim", "mlx", False, True),
        ("none", "mlx", True, True),
        ("cpu", "cpu", False, False),
    ]
    for w in a.widths:
        for name, dev, nl, no in variants:
            lat = measure(w, a.batch, dev, nl, no, a.iters)
            out.write({"width": w, "batch": a.batch, "variant": name, "latency": lat})
            print(
                f"w={w:<5} {name:<16} {lat['across_median_ms']:9.2f} ms (+/-{lat['across_std_ms']:.2f})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
