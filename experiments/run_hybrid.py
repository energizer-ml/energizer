#!/usr/bin/env python
"""Step 6 -- hybrid prototype: forward on Core ML (ANE), backward + optimizer on MLX.

Question: does the cost of moving data and of *refreshing the Core ML weights*
cancel the forward-pass gain? Core ML ML Programs are immutable: after every
optimizer step the new weights only reach the ANE through
trace -> convert -> save -> compile -> load. That "weight sync" is timed
separately and amortised over ``--sync-every K`` steps (K>1 trains with stale
forward weights, so it is a cost model, not a convergence claim).

Model: 3-layer ReLU MLP, cross-entropy, SGD. The Core ML graph returns all
post-ReLU activations so the GPU backward needs no recomputation.

    python experiments/run_hybrid.py --width 1024 --batch 256
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ezbench import coreml_utils as cu
from ezbench import timing
from ezbench.results import ResultWriter


def make_params(width: int, classes: int, seed: int = 0):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    dims = [(width, width), (width, width), (classes, width)]
    ws = [mx.array((rng.standard_normal(d) * 0.02).astype(np.float32)) for d in dims]
    bs = [mx.zeros((d[0],)) for d in dims]
    return ws, bs


def sync_coreml(ws, bs, example, precision: str, units: str):
    """Rebuild the Core ML model from MLX weights; return (model, seconds_by_stage)."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.ModuleList(nn.Linear(w.shape[1], w.shape[0]) for w in ws)

        def forward(self, x):
            a1 = torch.relu(self.l[0](x))
            a2 = torch.relu(self.l[1](a1))
            return a1, a2, self.l[2](a2)

    net = Net().eval()
    with torch.no_grad():
        for lin, w, b in zip(net.l, ws, bs):
            lin.weight.copy_(torch.from_numpy(np.array(w)))
            lin.bias.copy_(torch.from_numpy(np.array(b)))
    t = {}
    t0 = time.perf_counter()
    mlmodel, pkg = cu.convert(net, example, precision, units)
    t["convert_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plan = cu.placement(pkg, units)  # also compiles the package
    t["compile_plan_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = cu.load(pkg, units)
    t["load_s"] = time.perf_counter() - t0
    model._plan = plan
    model._pkg_owner = mlmodel  # keep the temp dir alive
    return model, t


def gpu_backward_sgd(ws, bs, x, a1, a2, logits, y, lr):
    """Manual MLP backward + SGD on MLX. Returns the loss (lazy) and new params."""
    import mlx.core as mx

    B = x.shape[0]
    probs = mx.softmax(logits, axis=1)
    onehot = (mx.arange(logits.shape[1])[None, :] == y[:, None]).astype(mx.float32)
    loss = -mx.mean(mx.sum(onehot * mx.log(probs + 1e-9), axis=1))
    g3 = (probs - onehot) / B
    gw3, gb3 = g3.T @ a2, g3.sum(0)
    g2 = (g3 @ ws[2]) * (a2 > 0)
    gw2, gb2 = g2.T @ a1, g2.sum(0)
    g1 = (g2 @ ws[1]) * (a1 > 0)
    gw1, gb1 = g1.T @ x, g1.sum(0)
    gws, gbs = [gw1, gw2, gw3], [gb1, gb2, gb3]
    return (
        loss,
        [w - lr * g for w, g in zip(ws, gws)],
        [b - lr * g for b, g in zip(bs, gbs)],
    )


def pure_mlx_step_fn(ws, bs, x, y, lr):
    import mlx.core as mx

    state = {"ws": ws, "bs": bs}

    def step():
        w, b = state["ws"], state["bs"]
        a1 = mx.maximum(x @ w[0].T + b[0], 0)
        a2 = mx.maximum(a1 @ w[1].T + b[1], 0)
        logits = a2 @ w[2].T + b[2]
        loss, nw, nb = gpu_backward_sgd(w, b, x, a1, a2, logits, y, lr)
        state["ws"], state["bs"] = nw, nb
        mx.eval(loss, *nw, *nb)

    return step


def main() -> int:
    import mlx.core as mx

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--units", default="ALL", choices=cu.COMPUTE_UNITS)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--syncs", type=int, default=3, help="weight syncs to time")
    ap.add_argument("--sync-every", type=int, nargs="+", default=[1, 10, 100, 1000])
    a = ap.parse_args()

    out = ResultWriter("hybrid")
    print("writing", out.path)
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((a.batch, a.width), dtype=np.float32)
    y_np = rng.integers(0, 10, a.batch).astype(np.int32)
    x, y = mx.array(x_np), mx.array(y_np)
    ws, bs = make_params(a.width, 10)
    lr = 0.01

    # 1) weight sync cost (convert + compile + load) -- the recompilation overhead
    import torch

    example = torch.from_numpy(x_np)
    syncs = []
    for _ in range(a.syncs):
        model, t = sync_coreml(ws, bs, example, a.precision, a.units)
        syncs.append(t)
    sync_s = float(np.median([sum(t.values()) for t in syncs]))
    plan = model._plan

    # 2) steady-state hybrid step components (weights fixed)
    def fwd_ane():
        return model.predict({"x": x_np})

    t_fwd = timing.measure(fwd_ane, iters=a.iters)
    outs = fwd_ane()
    # Core ML renames/reorders outputs: identify each by matching a NumPy reference.
    w_np, b_np = [np.array(w) for w in ws], [np.array(b) for b in bs]
    r1 = np.maximum(x_np @ w_np[0].T + b_np[0], 0)
    r2 = np.maximum(r1 @ w_np[1].T + b_np[1], 0)
    r3 = r2 @ w_np[2].T + b_np[2]
    keyed = {}
    for name, ref in (("a1", r1), ("a2", r2), ("logits", r3)):
        cands = [k for k in outs if outs[k].shape == ref.shape]
        keyed[name] = min(cands, key=lambda k: float(np.abs(outs[k] - ref).max()))
    assert len({*keyed.values()}) == 3, f"ambiguous Core ML outputs: {keyed}"
    a1_np, a2_np, lg_np = (outs[keyed[n]] for n in ("a1", "a2", "logits"))

    t_xfer = timing.measure(
        lambda: mx.eval(
            *[mx.array(np.asarray(v, dtype=np.float32)) for v in (a1_np, a2_np, lg_np)]
        ),
        iters=a.iters,
    )
    a1, a2, lg = (
        mx.array(np.asarray(v, dtype=np.float32)) for v in (a1_np, a2_np, lg_np)
    )

    def bwd_gpu():
        loss, nw, nb = gpu_backward_sgd(ws, bs, x, a1, a2, lg, y, lr)
        mx.eval(loss, *nw, *nb)

    t_bwd = timing.measure(bwd_gpu, iters=a.iters)

    # 3) baseline: everything on MLX
    t_pure = timing.measure(pure_mlx_step_fn(ws, bs, x, y, lr), warmup=5, iters=a.iters)

    rec = {
        "width": a.width,
        "batch": a.batch,
        "precision": a.precision,
        "units": a.units,
        "forward_coreml": t_fwd,
        "transfer_to_gpu": t_xfer,
        "backward_opt_gpu": t_bwd,
        "pure_mlx_step": t_pure,
        "weight_sync_s": sync_s,
        "weight_sync_stages": syncs,
        "placement_frac_ops": plan["frac_ops"],
    }
    out.write(rec)

    step_ms = t_fwd["median_ms"] + t_xfer["median_ms"] + t_bwd["median_ms"]
    print(f"placement (planned)  : {plan['frac_ops']}")
    if plan["frac_ops"].get("ANE", 0.0) < 0.5:
        print(
            "WARNING: <50% of ops planned on the ANE; 'Core ML forward' is not an ANE measurement"
        )
    print(f"Core ML forward      : {t_fwd['median_ms']:.2f} ms")
    print(f"transfer Core ML->GPU: {t_xfer['median_ms']:.2f} ms")
    print(f"GPU backward+SGD     : {t_bwd['median_ms']:.2f} ms")
    print(f"pure MLX step        : {t_pure['median_ms']:.2f} ms")
    print(f"weight sync (compile): {sync_s * 1e3:.0f} ms")
    for k in a.sync_every:
        amort = step_ms + sync_s * 1e3 / k
        verdict = "wins" if amort < t_pure["median_ms"] else "loses"
        print(
            f"  sync every {k:>4}: hybrid {amort:9.2f} ms/step  -> {verdict} vs pure MLX"
        )
        out.write(
            {
                "record": "amortised",
                "width": a.width,
                "sync_every": k,
                "hybrid_ms": amort,
                "pure_mlx_ms": t_pure["median_ms"],
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
