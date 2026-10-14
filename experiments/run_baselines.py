#!/usr/bin/env python
"""Step 4 -- same workloads on MLX, PyTorch (MPS / CPU) and Core ML.

python experiments/run_baselines.py --preset smoke
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import signal
import sys

import numpy as np

from ezbench import coreml_utils as cu
from ezbench import models, timing
from ezbench.results import RESULTS_DIR, ResultWriter, load
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


class RowTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise RowTimeout()


def _done_keys():
    """(workload, precision, framework) already measured in earlier runs."""
    keys = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, "baselines", "*.jsonl")):
        _, rows = load(f)
        for r in rows:
            if "error" not in r or r["error"].startswith("RowTimeout"):
                keys.add((r["workload"], r["precision"], r["framework"]))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--row-timeout", type=int, default=240, help="seconds per row")
    ap.add_argument(
        "--resume", action="store_true", help="skip rows found in earlier result files"
    )
    args = ap.parse_args()
    signal.signal(signal.SIGALRM, _alarm)
    done = _done_keys() if args.resume else set()

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
            if (models.name(spec), prec, name) in done:
                continue
            if prec == "fp16" and name == "torch_cpu":
                continue  # half-precision CPU kernels are not a meaningful baseline
            if prec == "fp32" and name == "coreml_cpu_ne":
                continue  # fp32 is never placed on the ANE (placement sweep): same as CPU
            rec = {
                "workload": models.name(spec),
                "spec": spec,
                "precision": prec,
                "framework": name,
                "flops": models.flops(spec),
            }
            signal.alarm(args.row_timeout)
            try:
                rec["latency"] = run(spec, prec)
            except RowTimeout:
                rec["error"] = "RowTimeout"
            except Exception as e:
                rec["error"] = repr(e)
            finally:
                signal.alarm(0)
            out.write(rec)
            med = rec.get("latency", {}).get("median_ms", float("nan"))
            print(f"{rec['workload']:<34} {prec} {name:<15} {med:9.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
