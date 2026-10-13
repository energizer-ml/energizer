#!/usr/bin/env python
"""Derive paper tables from raw results. Reads only ``results/**/*.jsonl``.

python experiments/make_tables.py            # latest file of each experiment
python experiments/make_tables.py --all      # merge every run
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from statistics import median

from ezbench.results import RESULTS_DIR, load

OUT = os.path.join(os.path.dirname(RESULTS_DIR), "paper", "tables")


def _files(exp: str, merge_all: bool):
    fs = sorted(glob.glob(os.path.join(RESULTS_DIR, exp, "*.jsonl")))
    return fs if merge_all else fs[-1:]


def _rows(exp: str, merge_all: bool):
    for f in _files(exp, merge_all):
        _, rows = load(f)
        yield from (r for r in rows if "error" not in r)


def _tex(path: str, header, body, caption: str, label: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    cols = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]\\centering",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}\\toprule",
        " & ".join(header) + " \\\\\\midrule",
        *[" & ".join(str(c) for c in r) + " \\\\" for r in body],
        "\\bottomrule\\end{tabular}\\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def placement_table(merge_all: bool) -> None:
    acc = defaultdict(list)
    for r in _rows("placement_sweep", merge_all):
        if r.get("compute_units") != "ALL" or "placement" not in r:
            continue
        key = (r["spec"]["kind"], r["precision"])
        acc[key].append(r["placement"]["frac_ops"].get("ANE", 0.0))
    body = [
        (
            k,
            p,
            len(v),
            f"{100 * median(v):.0f}\\%",
            f"{100 * min(v):.0f}\\%",
            f"{100 * max(v):.0f}\\%",
        )
        for (k, p), v in sorted(acc.items())
    ]
    _tex(
        os.path.join(OUT, "placement_ane_fraction.tex"),
        ["workload", "precision", "n", "median", "min", "max"],
        body,
        "Fraction of ops placed on the ANE by Core ML (compute units ALL).",
        "tab:ane-fraction",
    )


def baseline_table(merge_all: bool) -> None:
    acc = defaultdict(list)
    for r in _rows("baselines", merge_all):
        acc[(r["spec"]["kind"], r["precision"], r["framework"])].append(
            r["latency"]["median_ms"]
        )
    body = [
        (k, p, fw, len(v), f"{median(v):.3f}") for (k, p, fw), v in sorted(acc.items())
    ]
    _tex(
        os.path.join(OUT, "baselines_latency.tex"),
        ["workload", "precision", "framework", "n", "median ms"],
        body,
        "Median inference latency per framework.",
        "tab:baselines",
    )


def efficiency_table(merge_all: bool) -> None:
    best, rows_ = defaultdict(float), []
    for r in _rows("peak", merge_all):
        best[(r["backend"], r["precision"])] = max(
            best[(r["backend"], r["precision"])], r["tflops"]
        )
    for (be, pr), pk in sorted(best.items()):
        rows_.append((be.replace("_", "\\_"), pr, f"{pk:.2f}"))
    if rows_:
        _tex(
            os.path.join(OUT, "measured_peak.tex"),
            ["backend", "precision", "peak TFLOP/s"],
            rows_,
            "Best measured square-matmul throughput (denominator for achieved-vs-peak).",
            "tab:peak",
        )


def hybrid_table(merge_all: bool) -> None:
    body = []
    for r in _rows("hybrid", merge_all):
        if "forward_coreml" not in r:
            continue
        body.append(
            (
                r["width"],
                r["units"].replace("_", "\\_"),
                f"{r['forward_coreml']['median_ms']:.2f}",
                f"{r['transfer_to_gpu']['median_ms']:.2f}",
                f"{r['backward_opt_gpu']['median_ms']:.2f}",
                f"{r['pure_mlx_step']['median_ms']:.2f}",
                f"{r['weight_sync_s'] * 1e3:.0f}",
            )
        )
    if body:
        _tex(
            os.path.join(OUT, "hybrid_overhead.tex"),
            ["width", "units", "fwd", "xfer", "bwd+opt", "pure MLX", "sync (ms)"],
            body,
            "Hybrid step overhead breakdown (ms).",
            "tab:hybrid",
        )


def convergence_table(merge_all: bool) -> None:
    body = []
    for r in _rows("convergence", merge_all):
        h = r["history"][-1]
        body.append(
            (
                r["path"].replace("_", "\\_"),
                f"{h['loss']:.4f}",
                f"{100 * h['test_acc']:.2f}\\%",
                f"{h['train_s']:.1f}",
            )
        )
    if body:
        _tex(
            os.path.join(OUT, "convergence.tex"),
            ["path", "final loss", "test acc", "train s"],
            body,
            "Final loss, accuracy and training time per path (same seed and data).",
            "tab:convergence",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    placement_table(a.all)
    baseline_table(a.all)
    efficiency_table(a.all)
    hybrid_table(a.all)
    convergence_table(a.all)


if __name__ == "__main__":
    main()
