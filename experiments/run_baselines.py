#!/usr/bin/env python
"""Step 4 -- same workloads on MLX, PyTorch (MPS / CPU) and Core ML.

python experiments/run_baselines.py --preset smoke
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from ezbench import coreml_utils as cu
from ezbench import models, timing
from ezbench.results import ResultWriter
from run_placement_sweep import sweep


def run_mlx(spec, dtype: str, iters: int):
    import mlx.core as mx

    fn, x = models.build_mlx(spec)
    if dtype == "fp16":
        from mlx.utils import tree_map

        fn.update(tree_map(lambda p: p.astype(mx.float16), fn.parameters()))
        x = x.astype(mx.float16)
    mx.eval(fn.parameters(), x)
    return timing.measure(lambda: mx.eval(fn(x)), iters=iters)


def run_torch(spec, dtype: str, device: str, iters: int):
    import torch

    module, x = models.build_torch(spec)
    dt = torch.float16 if dtype == "fp16" else torch.float32
    module, x = module.to(device=device, dtype=dt), x.to(device=device, dtype=dt)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    def fn():
        with torch.no_grad():
            module(x)
        sync()

    return timing.measure(fn, iters=iters)


def run_coreml(spec, dtype: str, units: str, iters: int):
    module, example = models.build_torch(spec)
    mlmodel, pkg = cu.convert(module, example, dtype)
    m = cu.load(pkg, units)
    return timing.measure(
        cu.predict_fn(m, example.numpy().astype(np.float32)), iters=iters
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    out = ResultWriter("baselines")
    print("writing", out.path)
    runners = [
        ("mlx", lambda s, p: run_mlx(s, p, args.iters)),
        ("torch_mps", lambda s, p: run_torch(s, p, "mps", args.iters)),
        ("torch_cpu", lambda s, p: run_torch(s, p, "cpu", args.iters)),
        ("coreml_all", lambda s, p: run_coreml(s, p, "ALL", args.iters)),
        ("coreml_cpu_gpu", lambda s, p: run_coreml(s, p, "CPU_AND_GPU", args.iters)),
        ("coreml_cpu_ne", lambda s, p: run_coreml(s, p, "CPU_AND_NE", args.iters)),
    ]
    for spec, prec in itertools.product(sweep(args.preset), args.precisions):
        for name, run in runners:
            rec = {
                "workload": models.name(spec),
                "spec": spec,
                "precision": prec,
                "framework": name,
                "flops": models.flops(spec),
            }
            try:
                rec["latency"] = run(spec, prec)
            except Exception as e:
                rec["error"] = repr(e)
            out.write(rec)
            med = rec.get("latency", {}).get("median_ms", float("nan"))
            print(f"{rec['workload']:<34} {prec} {name:<15} {med:9.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
