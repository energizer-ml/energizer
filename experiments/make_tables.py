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
        (k, p, fw.replace("_", "\\_"), len(v), f"{median(v):.3f}")
        for (k, p, fw), v in sorted(acc.items())
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
    # one row per path: prefer a run with measured power, else the latest
    best = {}
    for r in _rows("convergence", merge_all):
        if not r.get("history"):
            continue
        has_p = bool((r.get("power") or {}).get("net"))
        cur = best.get(r["path"])
        if cur is None or has_p or not cur[0]:
            best[r["path"]] = (has_p, r)
    body = []
    for path, (_, r) in sorted(best.items()):
        h = r["history"][-1]
        pw_ = r.get("power") or {}
        pr = pw_.get("net_per_run_mj") or (
            {k: v / r.get("reps", 1) for k, v in pw_["net"].items()}
            if pw_.get("net")
            else None
        )
        energy = f"{pr['combined_net_mj'] / 1e3:.1f}" if pr else "--"
        per_run_s = r["wall_s"] / r.get("reps", 1)
        body.append(
            (
                path.replace("_", "\\_"),
                f"{h['loss']:.4f}",
                f"{100 * h['test_acc']:.2f}\\%",
                f"{per_run_s:.1f}",
                energy,
            )
        )
    if body:
        _tex(
            os.path.join(OUT, "convergence.tex"),
            ["path", "final loss", "test acc", "time s", "energy J"],
            body,
            "Same seed and data: final loss, accuracy, wall time and idle-subtracted energy per training run.",
            "tab:convergence",
        )


def power_table(merge_all: bool) -> None:
    """Idle-subtracted power and energy per run by compute units (needs --power data)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in _rows("placement_sweep", merge_all):
        pn = r.get("power_net")
        if not pn or r.get("precision") != "fp16":
            continue
        key = (r["spec"]["kind"], r["compute_units"].replace("_", "\\_"))
        for rail in ("cpu", "gpu", "ane"):
            if f"{rail}_net_mw" in pn:
                acc[key][rail].append(pn[f"{rail}_net_mw"])
        if r.get("energy_per_run_mj") is not None:
            acc[key]["mj"].append(r["energy_per_run_mj"])
    body = []
    for (kind, cu), d in sorted(acc.items()):
        f = lambda k, fmt="{:.0f}": fmt.format(median(d[k])) if d[k] else "--"
        body.append((kind, cu, f("cpu"), f("gpu"), f("ane"), f("mj", "{:.3f}")))
    if body:
        _tex(
            os.path.join(OUT, "power_by_units.tex"),
            ["workload", "units", "CPU mW", "GPU mW", "ANE mW", "mJ/run"],
            body,
            "Median idle-subtracted power per rail and energy per inference (fp16).",
            "tab:power",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--latest", action="store_true", help="only the newest file per experiment"
    )
    a = ap.parse_args()
    a.all = not a.latest  # default: merge every run (resumed sweeps span files)
    placement_table(a.all)
    baseline_table(a.all)
    efficiency_table(a.all)
    hybrid_table(a.all)
    convergence_table(a.all)
    power_table(a.all)


if __name__ == "__main__":
    main()
