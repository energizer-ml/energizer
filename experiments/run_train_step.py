#!/usr/bin/env python
"""Step 5 -- training-step latency of Energizer on CPU (NumPy) vs MLX.

Times ``zero_grad -> forward -> loss -> backward -> optimizer.step`` and forces
all lazy MLX work to finish (``mx.eval`` on parameters and loss) so the number
is the true step time, not graph-construction time.

    python experiments/run_train_step.py --tag before
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import energizer as ez
from ezbench import timing
from ezbench.results import ResultWriter

WORKLOADS = {
    "mlp_1024x3": dict(kind="mlp", batch=256, width=1024),
    "mlp_2048x3": dict(kind="mlp", batch=256, width=2048),
    "cnn_c32_hw32": dict(kind="cnn", batch=64, channels=32, hw=32),
    "tfm_s64_d256_l2": dict(kind="tfm", batch=16, seq=64, d_model=256, layers=2),
}


def build(cfg: dict, device: str):
    """Return (model, input_np, n_classes)."""
    rng = np.random.default_rng(0)
    kind = cfg["kind"]
    if kind == "mlp":
        w = cfg["width"]
        model = ez.Sequential(
            ez.Linear(w, w), ez.ReLU(), ez.Linear(w, w), ez.ReLU(), ez.Linear(w, 10)
        )
        x = rng.standard_normal((cfg["batch"], w), dtype=np.float32)
    elif kind == "cnn":
        c, hw = cfg["channels"], cfg["hw"]
        model = ez.Sequential(
            ez.Conv2d(3, c, 3, padding=1),
            ez.BatchNorm2d(c),
            ez.ReLU(),
            ez.MaxPool2d(2, stride=2),
            ez.Conv2d(c, c, 3, padding=1),
            ez.ReLU(),
            ez.AvgPool2d(2, stride=2),
            ez.Flatten(),
            ez.Linear(c * (hw // 4) ** 2, 10),
        )
        x = rng.standard_normal((cfg["batch"], 3, hw, hw), dtype=np.float32)
    elif kind == "tfm":
        d = cfg["d_model"]
        layer = ez.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.0, batch_first=True, activation="gelu"
        )
        model = ez.Sequential(
            ez.TransformerEncoder(layer, cfg["layers"]),
            ez.Reshape((cfg["batch"], cfg["seq"] * d)),
            ez.Linear(cfg["seq"] * d, 10),
        )
        x = rng.standard_normal((cfg["batch"], cfg["seq"], d), dtype=np.float32)
    else:
        raise ValueError(kind)
    return model.to(device), x


def sync(device: str, params, loss) -> None:
    if device not in ("mlx", "gpu"):  # "gpu" = pre-rename alias
        return
    import mlx.core as mx

    mx.eval(loss.data, *[p.data for p in params])


def run(name: str, cfg: dict, device: str, optimizer: str, iters: int):
    model, x_np = build(cfg, device)
    model.train()
    params = model.parameters()
    opt = (ez.Adam if optimizer == "adam" else ez.SGD)(params, lr=1e-3)
    lf = ez.CrossEntropyLoss()
    rng = np.random.default_rng(1)
    x = ez.Tensor(x_np, device=device)
    y = ez.Tensor(rng.integers(0, 10, cfg["batch"]).astype(np.float32), device=device)

    def step():
        opt.zero_grad()
        loss = lf(model(x), y)
        loss.backward()
        opt.step()
        sync(device, params, loss)

    return timing.measure(step, warmup=3, iters=iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--devices", nargs="+", default=["cpu", "mlx"])
    ap.add_argument("--optimizers", nargs="+", default=["adam", "sgd"])
    a = ap.parse_args()
    out = ResultWriter("train_step")
    print("writing", out.path)
    for name in a.workloads:
        cfg = WORKLOADS[name]
        for dev in a.devices:
            for opt in a.optimizers:
                lat = run(name, cfg, dev, opt, a.iters)
                out.write(
                    {
                        "workload": name,
                        "cfg": cfg,
                        "device": dev,
                        "optimizer": opt,
                        "tag": a.tag,
                        "latency": lat,
                    }
                )
                print(f"{name:<12} {dev:<4} {opt:<5} {lat['median_ms']:8.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
