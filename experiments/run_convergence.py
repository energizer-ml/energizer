#!/usr/bin/env python
"""Does the accelerated path train to the same accuracy as the fp32 baseline?

Same model, data, seed and optimizer on four paths:

  cpu          Energizer NumPy fp32           (reference)
  mlx          Energizer MLX fp32 (GPU)
  hybrid_fp16  forward in Core ML fp16 (``--units``, ANE when placed there),
               backward + SGD on MLX; Core ML weights refreshed every
               ``--sync-every`` steps, so the forward uses stale weights between
               syncs (the cost model of ``run_hybrid.py`` made real).

Records loss / test accuracy per epoch with wall time, so time-to-accuracy and
(with ``--power`` under sudo) energy-to-accuracy can be derived.

    python experiments/run_convergence.py --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import energizer as ez
from ezbench import power as pw
from ezbench.power import PowerSampler
from ezbench.results import ResultWriter


def load_mnist(n_train: int, n_test: int):
    """MNIST via torchvision (downloaded to ~/.cache/energizer-data)."""
    import os

    from torchvision import datasets

    root = os.path.expanduser("~/.cache/energizer-data")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds, n):
        x = ds.data[:n].numpy().astype(np.float32) / 255.0
        return ((x - 0.1307) / 0.3081).reshape(len(x), -1), ds.targets[:n].numpy()

    return prep(tr, n_train), prep(te, n_test)


def init_params(width: int, seed: int):
    rng = np.random.default_rng(seed)
    dims = [(width, 784), (width, width), (10, width)]
    ws = [
        (rng.standard_normal(d) * np.sqrt(2.0 / d[1])).astype(np.float32) for d in dims
    ]
    bs = [np.zeros(d[0], np.float32) for d in dims]
    return ws, bs


def eval_np(ws, bs, x, y):
    h = np.maximum(x @ ws[0].T + bs[0], 0)
    h = np.maximum(h @ ws[1].T + bs[1], 0)
    lg = h @ ws[2].T + bs[2]
    return float((lg.argmax(1) == y).mean())


def run_energizer(device, data, a):
    (xtr, ytr), (xte, yte) = data
    ws, bs = init_params(a.width, a.seed)
    model = ez.Sequential(
        ez.Linear(784, a.width),
        ez.ReLU(),
        ez.Linear(a.width, a.width),
        ez.ReLU(),
        ez.Linear(a.width, 10),
    )
    lins = [m for m in model.layers if isinstance(m, ez.Linear)]
    for lin, w, b in zip(lins, ws, bs):
        lin.weight.data, lin.bias.data = w.copy(), b.copy()
    model.to(device)
    opt = ez.SGD(model.parameters(), lr=a.lr)
    lf = ez.CrossEntropyLoss()
    rng = np.random.default_rng(a.seed)
    hist, t_train = [], 0.0
    for ep in range(a.epochs):
        order = rng.permutation(len(xtr))
        t0 = time.perf_counter()
        losses = []
        for i in range(0, len(order) - a.batch + 1, a.batch):
            idx = order[i : i + a.batch]
            x = ez.Tensor(xtr[idx], device=device)
            y = ez.Tensor(ytr[idx].astype(np.float32), device=device)
            opt.zero_grad()
            loss = lf(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss)
        if device == "mlx":
            import mlx.core as mx

            mx.eval([p.data for p in model.parameters()])
        t_train += time.perf_counter() - t0
        mean_loss = float(np.mean([float(np.array(l.data)) for l in losses]))
        wn = [np.array(m.weight.data) for m in lins]
        bn = [np.array(m.bias.data) for m in lins]
        hist.append(
            {
                "epoch": ep + 1,
                "loss": mean_loss,
                "test_acc": eval_np(wn, bn, xte, yte),
                "train_s": t_train,
            }
        )
    return hist


def run_hybrid(data, a, sync_every):
    import mlx.core as mx
    import torch

    from run_hybrid import gpu_backward_sgd, sync_coreml

    (xtr, ytr), (xte, yte) = data
    ws_np, bs_np = init_params(a.width, a.seed)
    ws, bs = [mx.array(w) for w in ws_np], [mx.array(b) for b in bs_np]
    example = torch.from_numpy(xtr[: a.batch])
    rng = np.random.default_rng(a.seed)
    model, _ = sync_coreml(ws, bs, example, "fp16", a.units)
    step, hist, t_train = 0, [], 0.0
    sync_total = 0.0
    for ep in range(a.epochs):
        order = rng.permutation(len(xtr))
        t0 = time.perf_counter()
        losses = []
        for i in range(0, len(order) - a.batch + 1, a.batch):
            idx = order[i : i + a.batch]
            xb = xtr[idx]
            outs = model.predict({"x": xb})
            a1, a2, lg = _assign(outs, xb, ws, bs, a.width)
            loss, ws, bs = gpu_backward_sgd(
                ws,
                bs,
                mx.array(xb),
                mx.array(a1),
                mx.array(a2),
                mx.array(lg),
                mx.array(ytr[idx].astype(np.int32)),
                a.lr,
            )
            mx.eval(loss, *ws, *bs)
            losses.append(float(loss))
            step += 1
            if step % sync_every == 0:
                ts = time.perf_counter()
                model, _ = sync_coreml(ws, bs, example, "fp16", a.units)
                sync_total += time.perf_counter() - ts
        t_train += time.perf_counter() - t0
        wn, bn = [np.array(w) for w in ws], [np.array(b) for b in bs]
        hist.append(
            {
                "epoch": ep + 1,
                "loss": float(np.mean(losses)),
                "test_acc": eval_np(wn, bn, xte, yte),
                "train_s": t_train,
                "sync_s": sync_total,
            }
        )
    return hist


def _assign(outs, xb, ws, bs, width):
    """Map Core ML outputs to (a1, a2, logits) by closeness to a float32 reference."""
    r1 = np.maximum(xb @ np.array(ws[0]).T + np.array(bs[0]), 0)
    r2 = np.maximum(r1 @ np.array(ws[1]).T + np.array(bs[1]), 0)
    r3 = r2 @ np.array(ws[2]).T + np.array(bs[2])
    got = {}
    for name, ref in (("a1", r1), ("a2", r2), ("lg", r3)):
        cands = [k for k in outs if outs[k].shape == ref.shape]
        got[name] = min(cands, key=lambda k: float(np.abs(outs[k] - ref).max()))
    return outs[got["a1"]], outs[got["a2"]], outs[got["lg"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--sync-every", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--paths", nargs="+", default=["cpu", "mlx", "hybrid_fp16"])
    ap.add_argument("--power", action="store_true")
    ap.add_argument(
        "--min-power-s",
        type=float,
        default=10.0,
        help="min power window for short paths",
    )
    a = ap.parse_args()

    if a.power and not PowerSampler.available():
        print("error: --power needs root", file=sys.stderr)
        return 2
    data = load_mnist(a.n_train, a.n_test)
    out = ResultWriter("convergence")
    print("writing", out.path)
    idle = pw.idle_baseline() if a.power else None
    jobs = []
    for path in a.paths:
        if path == "hybrid_fp16":
            jobs += [(f"hybrid_fp16_k{k}", k) for k in a.sync_every]
        else:
            jobs.append((path, None))
    for path, k in jobs:
        ctx = PowerSampler() if a.power else None
        t0 = time.perf_counter()
        if ctx:
            ctx.__enter__()
        status = "ok"
        reps = 1
        try:
            if k is not None:
                hist = run_hybrid(data, a, k)
            else:
                # short trainings are repeated so the power window holds enough
                # powermetrics samples; energy is reported per training run
                device = "mlx" if path == "mlx" else "cpu"
                hist = run_energizer(device, data, a)
                while a.power and time.perf_counter() - t0 < a.min_power_s:
                    hist = run_energizer(device, data, a)
                    reps += 1
        except (
            AssertionError
        ) as e:  # coremltools rejects NaN weights: training diverged
            hist, status = [], f"diverged: {e}"
        wall = time.perf_counter() - t0
        power = None
        if ctx:
            ctx.__exit__(None, None, None)
            net = pw.net(ctx.summary(), idle, wall)
            per_run = (
                {k_: v / reps for k_, v in net.items() if k_.endswith("_mj")}
                if net
                else None
            )
            power = {"summary": ctx.summary(), "net": net, "net_per_run_mj": per_run}
        rec = {
            "path": path,
            "sync_every": k,
            "args": vars(a),
            "history": hist,
            "wall_s": wall,
            "reps": reps,
            "power": power,
            "status": status,
        }
        if status != "ok":
            rec["error"] = status
        out.write(rec)
        if hist:
            h = hist[-1]
            print(
                f"{path:<16} loss {h['loss']:.4f}  test_acc {h['test_acc']:.4f}  train {h['train_s']:.1f}s"
            )
        else:
            print(f"{path:<16} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
