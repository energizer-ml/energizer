#!/usr/bin/env python
"""Frozen-trunk fine-tuning: the one training use case where the ANE forward is static.

A trunk (MLP or CNN) is pre-trained on MNIST digits 0-4 and then *frozen*. A new
linear head is trained on digits 5-9 (transfer task). Because the trunk never
changes, its Core ML model is compiled once -- no weight sync, which is what
sank the hybrid design of ``run_hybrid.py``.

Paths (all train the same head with the same batches and Adam):

  mlx_fp32 / mlx_fp16   trunk forward on the MLX GPU, every step
  coreml_<UNITS>        trunk forward in Core ML fp16 (CPU_AND_NE = ANE), every step
  coreml_<UNITS>_pipe   same, but batch i+1 is predicted in a thread while the
                        GPU trains the head on batch i (ANE/GPU overlap)
  *_cached              trunk features extracted once, head trained on the cache
                        (the amortised case: epochs >> 1)

Metrics: wall time (median of ``--repeats``), samples/s, head test accuracy, and
with ``--power`` (run under sudo) idle-subtracted energy per run per rail.

    python experiments/run_frozen.py --preset smoke
    sudo -E python experiments/run_frozen.py --preset full --power
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ezbench import coreml_utils as cu
from ezbench import power as pw
from ezbench.power import PowerSampler
from ezbench.results import ResultWriter

HEAD_CLASSES = 5


# ----------------------------------------------------------------------- data
def load_split():
    """MNIST split into a pre-training task (0-4) and a transfer task (5-9)."""
    from torchvision import datasets

    root = os.path.expanduser("~/.cache/energizer-data")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.numpy().astype(np.float32) / 255.0
        return ((x - 0.1307) / 0.3081).reshape(len(x), 784), ds.targets.numpy()

    (xa, ya), (xb, yb) = prep(tr), prep(te)
    pre = ya < HEAD_CLASSES
    tra, teb = ya >= HEAD_CLASSES, yb >= HEAD_CLASSES
    return {
        "pre": (xa[pre], ya[pre]),
        "train": (xa[tra], ya[tra] - HEAD_CLASSES),
        "test": (xb[teb], yb[teb] - HEAD_CLASSES),
    }


# ---------------------------------------------------------------------- trunk
CNN_PLAN = [(1, 1, 1), (1, 1, 1), (1, 2, 2), (2, 2, 1), (2, 4, 2), (4, 4, 1)]


def build_trunk(arch: str, width: int, depth: int):
    import torch
    import torch.nn as nn

    if arch == "mlp":

        class Trunk(nn.Module):
            def __init__(self):
                super().__init__()
                dims = [784] + [width] * depth
                self.layers = nn.ModuleList(
                    nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
                )
                self.feat = width

            def forward(self, x):
                for lin in self.layers:
                    x = torch.relu(lin(x))
                return x

        return Trunk(), (784,)

    plan = CNN_PLAN[:depth]

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.strides = [s for _, _, s in plan]
            self.layers = nn.ModuleList(
                nn.Conv2d(
                    1 if i == 0 else plan[i - 1][1] * width, o * width, 3, s, padding=1
                )
                for i, (_, o, s) in enumerate(plan)
            )
            # the plan's channel multipliers apply to ``width`` (first conv is 1->width)
            self.feat = plan[-1][1] * width

        def forward(self, x):
            for conv in self.layers:
                x = torch.relu(conv(x))
            return x.mean(dim=(2, 3))

    return Trunk(), (1, 28, 28)


def pretrain(trunk, data, steps_epochs: int, seed: int):
    """Supervised pre-training on digits 0-4 (torch, MPS if present)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x, y = data["pre"]
    shape = (-1,) + trunk._in_shape
    head = nn.Linear(trunk.feat, HEAD_CLASSES)
    net = nn.Sequential(trunk, head).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps_epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(order) - 127, 128):
            idx = order[i : i + 128]
            xb = torch.from_numpy(x[idx]).reshape(shape).to(dev)
            yb = torch.from_numpy(y[idx]).long().to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(net(xb), yb).backward()
            opt.step()
    trunk.cpu().eval()
    for p in trunk.parameters():
        p.requires_grad_(False)


def mlx_trunk_fn(trunk, arch: str, dtype):
    """MLX forward built from the torch trunk's weights (same function, GPU)."""
    import mlx.core as mx

    ps = []
    for layer in trunk.layers:
        w = layer.weight.detach().numpy()
        if arch == "cnn":
            w = w.transpose(0, 2, 3, 1)  # OIHW -> OHWI (MLX is channels-last)
        ps.append((mx.array(w).astype(dtype), mx.array(layer.bias.detach().numpy()).astype(dtype)))
    strides = getattr(trunk, "strides", None)

    def fwd(x):  # x: (B, 784) float32
        x = x.astype(dtype)
        if arch == "mlp":
            for w, b in ps:
                x = mx.maximum(x @ w.T + b, 0)
            return x.astype(mx.float32)
        x = x.reshape(x.shape[0], 28, 28, 1)
        for (w, b), s in zip(ps, strides):
            x = mx.maximum(mx.conv2d(x, w, stride=s, padding=1) + b, 0)
        return x.mean(axis=(1, 2)).astype(mx.float32)

    return fwd


def coreml_trunk(trunk, in_shape, batch: int, units: str):
    """Convert the frozen trunk once; return (predict_fn, placement)."""
    import torch

    example = torch.randn((batch,) + in_shape)
    mlmodel, pkg = cu.convert(trunk, example, "fp16", units)
    plan = cu.placement(pkg, units)
    model = cu.load(pkg, units)
    model._owner = mlmodel  # keep temp dir alive
    shape = (batch,) + in_shape

    def predict(xb: np.ndarray) -> np.ndarray:
        out = model.predict({"x": np.ascontiguousarray(xb.reshape(shape))})
        return np.asarray(next(iter(out.values())), dtype=np.float32)

    return predict, plan


# ------------------------------------------------------------------- training
def make_head(feat: int, seed: int):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    head = nn.Linear(feat, HEAD_CLASSES)
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(m, f, y):
        return nn.losses.cross_entropy(m(f), y, reduction="mean")

    vg = nn.value_and_grad(head, loss_fn)

    def step(f, y):
        loss, g = vg(head, f, y)
        opt.update(head, g)
        mx.eval(loss, head.parameters(), opt.state)
        return loss

    return head, step


def batches(n: int, batch: int, epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            yield order[i : i + batch]


def accuracy(head, feats_fn, x, y, batch: int) -> float:
    import mlx.core as mx

    n = len(x) // batch * batch
    correct = 0
    for i in range(0, n, batch):
        f = feats_fn(x[i : i + batch])
        pred = mx.argmax(head(mx.array(f) if isinstance(f, np.ndarray) else f), axis=1)
        correct += int((np.array(pred) == y[i : i + batch]).sum())
    return correct / n


def run_path(kind, feats_fn, data, a, feat, batch):
    """Train the head once. Returns (seconds, extract_seconds, test_acc)."""
    import mlx.core as mx

    xtr, ytr = data["train"]
    n = len(xtr) // batch * batch
    xtr, ytr = xtr[:n], ytr[:n]
    head, step = make_head(feat, a.seed)
    t0 = time.perf_counter()
    extract = 0.0
    to_mx = lambda f: mx.array(f) if isinstance(f, np.ndarray) else f

    if kind.endswith("_cached"):
        te = time.perf_counter()
        parts = [to_mx(feats_fn(xtr[i : i + batch])) for i in range(0, n, batch)]
        cache = mx.concatenate(parts, axis=0)
        mx.eval(cache)
        extract = time.perf_counter() - te
        ymx = mx.array(ytr.astype(np.int32))
        for idx in batches(n, batch, a.epochs, a.seed):
            ix = mx.array(idx)
            step(cache[ix], ymx[ix])
    elif kind.endswith("_pipe"):
        ex = ThreadPoolExecutor(max_workers=1)
        idxs = list(batches(n, batch, a.epochs, a.seed))
        fut = ex.submit(feats_fn, xtr[idxs[0]])
        for j, idx in enumerate(idxs):
            f = fut.result()
            if j + 1 < len(idxs):
                fut = ex.submit(feats_fn, xtr[idxs[j + 1]])
            step(to_mx(f), mx.array(ytr[idx].astype(np.int32)))
        ex.shutdown()
    else:
        for idx in batches(n, batch, a.epochs, a.seed):
            f = to_mx(feats_fn(xtr[idx]))
            step(f, mx.array(ytr[idx].astype(np.int32)))
    mx.eval(head.parameters())
    secs = time.perf_counter() - t0
    xte, yte = data["test"]
    return secs, extract, accuracy(head, feats_fn, xte, yte, batch)


def measured(fn, repeats):
    runs = [fn() for _ in range(repeats)]
    secs = [r[0] for r in runs]
    return {
        "median_s": statistics.median(secs),
        "min_s": min(secs),
        "max_s": max(secs),
        "runs_s": secs,
        "extract_s": statistics.median(r[1] for r in runs),
        "test_acc": runs[-1][2],
    }


def main() -> int:
    import mlx.core as mx

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--min-power-s", type=float, default=10.0)
    ap.add_argument("--units", nargs="+", default=["CPU_AND_NE", "ALL", "CPU_AND_GPU"])
    a = ap.parse_args()

    # (arch, width, depth, batch)
    if a.preset == "smoke":
        a.epochs, a.repeats = 1, 1
        configs = [("mlp", 256, 3, 128), ("cnn", 8, 4, 128)]
    else:
        configs = [
            ("mlp", 512, 4, 256),
            ("mlp", 1024, 6, 256),
            ("mlp", 2048, 8, 256),
            ("mlp", 2048, 8, 64),
            ("cnn", 16, 6, 256),
            ("cnn", 32, 6, 256),
            ("cnn", 64, 6, 256),
        ]
    if a.power and not PowerSampler.available():
        print("error: --power needs root (sudo -E)", file=sys.stderr)
        return 2

    out = ResultWriter("frozen")
    print("writing", out.path)
    if not out._env.get("on_ac"):
        print("WARNING: running on battery; plug in for clean power numbers")
    data = load_split()
    idle = pw.idle_baseline() if a.power else None

    for arch, width, depth, batch in configs:
        trunk, in_shape = build_trunk(arch, width, depth)
        trunk._in_shape = in_shape
        pretrain(trunk, data, 1 if a.preset == "smoke" else 2, a.seed)
        feat = trunk.feat
        n_params = sum(p.numel() for p in trunk.parameters())
        name = f"{arch}-w{width}-d{depth}-b{batch}"
        print(f"\n== {name}  ({n_params / 1e6:.1f}M trunk params, {feat}-d features)")

        paths = {}  # label -> feats_fn
        placements = {}
        for dt_name, dt in (("fp32", mx.float32), ("fp16", mx.float16)):
            fwd = mlx_trunk_fn(trunk, arch, dt)
            paths[f"mlx_{dt_name}"] = (
                lambda xb, fwd=fwd: (lambda f: (mx.eval(f), f)[1])(fwd(mx.array(xb)))
            )
        for units in a.units:
            try:
                predict, plan = coreml_trunk(trunk, in_shape, batch, units)
            except Exception as e:  # conversion failure is a result, not a crash
                out.write({"config": name, "path": f"coreml_{units}", "status": f"error: {e}"})
                print(f"coreml_{units:<12} conversion failed: {e}")
                continue
            paths[f"coreml_{units}"] = predict
            placements[units] = plan["frac_ops"]
            print(f"  placement {units}: {plan['frac_ops']}")

        # feature fidelity of the fp16 Core ML trunk vs the fp32 torch reference
        import torch

        xb = data["test"][0][:batch]
        ref = trunk(torch.from_numpy(xb).reshape((batch,) + in_shape)).numpy()
        fidelity = {}
        for label, fn in paths.items():
            f = np.array(fn(xb))
            fidelity[label] = float(np.abs(f - ref).max() / (np.abs(ref).max() + 1e-9))

        runs = []
        for label in list(paths):
            runs.append((label, label, paths[label]))
            runs.append((f"{label}_cached", f"{label}_cached", paths[label]))
            if label.startswith("coreml_"):
                runs.append((f"{label}_pipe", f"{label}_pipe", paths[label]))

        for label, kind, fn in runs:
            res = measured(lambda: run_path(kind, fn, data, a, feat, batch), a.repeats)
            steps = a.epochs * (len(data["train"][0]) // batch)
            res["ms_per_step"] = res["median_s"] * 1e3 / steps
            res["samples_per_s"] = steps * batch / res["median_s"]
            power = None
            if a.power:
                reps, t0 = 0, time.perf_counter()
                with PowerSampler() as ps:
                    while time.perf_counter() - t0 < a.min_power_s or reps == 0:
                        run_path(kind, fn, data, a, feat, batch)
                        reps += 1
                wall = time.perf_counter() - t0
                net = pw.net(ps.summary(), idle, wall)
                power = {
                    "reps": reps,
                    "wall_s": wall,
                    "summary": ps.summary(),
                    "net_per_run_mj": (
                        {k: v / reps for k, v in net.items() if k.endswith("_mj")}
                        if net
                        else None
                    ),
                }
            out.write(
                {
                    "config": name,
                    "arch": arch,
                    "width": width,
                    "depth": depth,
                    "batch": batch,
                    "trunk_params": n_params,
                    "feat_dim": feat,
                    "path": label,
                    "epochs": a.epochs,
                    "placement": placements,
                    "feature_rel_err": fidelity.get(label.split("_cached")[0].split("_pipe")[0]),
                    "timing": res,
                    "power": power,
                    "status": "ok",
                }
            )
            e = ""
            if power and power["net_per_run_mj"]:
                e = f"  {power['net_per_run_mj'].get('combined_net_mj', 0) / 1e3:7.2f} J"
            print(
                f"  {label:<26} {res['median_s']:7.2f} s  {res['ms_per_step']:7.2f} ms/step"
                f"  acc {res['test_acc']:.4f}{e}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
