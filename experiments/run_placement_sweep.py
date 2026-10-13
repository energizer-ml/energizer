#!/usr/bin/env python
"""Step 2/3 -- placement, latency and power for Core ML models.

For every (workload, precision) pair, convert once, ask ``MLComputePlan`` where
each op would run, then time real predictions under each compute-unit setting
(optionally with powermetrics sampling; run under ``sudo`` for power).

    python experiments/run_placement_sweep.py --preset smoke
    sudo -E python experiments/run_placement_sweep.py --preset full --power
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np

from ezbench import coreml_utils as cu
from ezbench import models, timing
from ezbench import power as pw
from ezbench.power import PowerSampler
from ezbench.results import ResultWriter


def sweep(preset: str):
    if preset == "smoke":
        yield {"kind": "conv", "batch": 1, "channels": 32, "hw": 32, "kernel": 3}
        yield {"kind": "matmul", "batch": 1, "m": 256, "k": 256, "n": 256}
        yield {
            "kind": "transformer",
            "batch": 1,
            "seq": 64,
            "d_model": 128,
            "heads": 4,
            "ffn_mult": 4,
        }
        return
    for b, c, hw in itertools.product(
        (1, 8), (16, 32, 64, 128, 256, 512), (32, 64, 128)
    ):
        yield {"kind": "conv", "batch": b, "channels": c, "hw": hw, "kernel": 3}
    for b, s in itertools.product((1, 8), (128, 256, 512, 1024, 2048, 4096)):
        yield {"kind": "matmul", "batch": b, "m": s, "k": s, "n": s}
    for b, s, d in itertools.product((1, 8), (16, 64, 128, 256, 512, 1024), (256, 512)):
        yield {
            "kind": "transformer",
            "batch": b,
            "seq": s,
            "d_model": d,
            "heads": d // 64,
            "ffn_mult": 4,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--units", nargs="+", default=list(cu.COMPUTE_UNITS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="independent timing runs (run-to-run spread)",
    )
    ap.add_argument(
        "--power", action="store_true", help="sample powermetrics (needs sudo)"
    )
    ap.add_argument("--power-seconds", type=float, default=5.0)
    args = ap.parse_args()

    if args.power and not PowerSampler.available():
        print("error: --power needs root (sudo) and powermetrics", file=sys.stderr)
        return 2

    idle = pw.idle_baseline() if args.power else None
    out = ResultWriter("placement_sweep")
    if idle:
        out.write({"record": "idle_baseline", "power": idle})
    print("writing", out.path)
    for spec, prec in itertools.product(sweep(args.preset), args.precisions):
        base = {
            "workload": models.name(spec),
            "spec": spec,
            "precision": prec,
            "flops": models.flops(spec),
        }
        try:
            module, example = models.build_torch(spec)
            mlmodel, pkg = cu.convert(module, example, prec)
        except Exception as e:  # conversion failures are data, not crashes
            out.write({**base, "stage": "convert", "error": repr(e)})
            continue
        x = example.numpy().astype(np.float32)
        for units in args.units:
            rec = {**base, "compute_units": units}
            try:
                rec["placement"] = cu.placement(pkg, units)
                m = cu.load(pkg, units)
                fn = cu.predict_fn(m, x)
                rec["latency"] = timing.repeat(
                    lambda: fn, repeats=args.repeats, warmup=10, iters=args.iters
                )
                if args.power:
                    with PowerSampler() as ps:
                        t_end = time.perf_counter() + args.power_seconds
                        n = 0
                        while time.perf_counter() < t_end:
                            fn()
                            n += 1
                    rec["power"] = ps.summary()
                    rec["power_net"] = pw.net(rec["power"], idle, args.power_seconds)
                    rec["power_runs"] = n
                    rec["energy_per_run_mj"] = (
                        rec["power_net"]["combined_net_mj"] / n
                        if rec["power_net"] and n
                        else None
                    )
            except Exception as e:
                rec["error"] = repr(e)
            out.write(rec)
            ane = rec.get("placement", {}).get("frac_ops", {}).get("ANE", 0.0)
            med = rec.get("latency", {}).get("median_ms", float("nan"))
            print(
                f"{base['workload']:<34} {prec} {units:<12} ANE={ane:5.0%}  {med:8.3f} ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
