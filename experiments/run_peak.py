#!/usr/bin/env python
"""Measured peak throughput per backend (denominator for "achieved vs peak").

Vendor peak numbers are not used: the peak is the best throughput *we* measure
for large square matmuls on each backend/precision. Efficiency of any other
workload is then ``achieved_tflops / measured_peak_tflops`` for the same
backend (``make_tables.py`` / ``make_figures.py``).

    python experiments/run_peak.py
"""

from __future__ import annotations

import argparse
import itertools
import sys

from ezbench import models
from ezbench.results import ResultWriter
from run_baselines import run_coreml, run_mlx, run_torch

SIZES = (512, 1024, 2048, 4096)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    out = ResultWriter("peak")
    print("writing", out.path)
    runners = {
        "coreml_cpu_ne": lambda s, p: run_coreml(s, p, "CPU_AND_NE", a.iters),
        "coreml_cpu_gpu": lambda s, p: run_coreml(s, p, "CPU_AND_GPU", a.iters),
        "mlx": lambda s, p: run_mlx(s, p, a.iters),
        "torch_mps": lambda s, p: run_torch(s, p, "mps", a.iters),
    }
    for n, prec in itertools.product(a.sizes, ("fp16", "fp32")):
        spec = {"kind": "matmul", "batch": 1, "m": n, "k": n, "n": n}
        for name, run in runners.items():
            rec = {
                "backend": name,
                "precision": prec,
                "size": n,
                "flops": models.flops(spec),
            }
            try:
                lat = run(spec, prec)
                rec["latency"] = lat
                rec["tflops"] = rec["flops"] / (lat["median_ms"] * 1e-3) / 1e12
            except Exception as e:
                rec["error"] = repr(e)
            out.write(rec)
            print(
                f"{name:<15} {prec} n={n:<5} {rec.get('tflops', float('nan')):7.2f} TFLOP/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
