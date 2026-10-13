#!/usr/bin/env python
"""Regenerate every paper figure from ``results/**/*.jsonl`` (no hand-edited data).

python experiments/make_figures.py [--all]    # writes paper/figures/*.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ezbench.results import RESULTS_DIR, load  # noqa: E402

OUT = os.path.join(os.path.dirname(RESULTS_DIR), "paper", "figures")


def rows(exp, merge_all):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, exp, "*.jsonl")))
    for f in files if merge_all else files[-1:]:
        _, rs = load(f)
        yield from (r for r in rs if "error" not in r)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def peaks(merge_all):
    best = defaultdict(float)
    for r in rows("peak", merge_all):
        best[(r["backend"], r["precision"])] = max(
            best[(r["backend"], r["precision"])], r["tflops"]
        )
    return best


def fig_ane_fraction(merge_all):
    """ANE-placed share of ops vs workload size, per workload kind and precision."""
    size = {
        "conv": lambda s: s["channels"],
        "matmul": lambda s: s["m"],
        "transformer": lambda s: s["seq"],
    }
    xlabel = {
        "conv": "channels",
        "matmul": "matrix size n",
        "transformer": "sequence length",
    }
    data = defaultdict(lambda: defaultdict(list))
    for r in rows("placement_sweep", merge_all):
        if r.get("compute_units") != "ALL" or "placement" not in r:
            continue
        k = r["spec"]["kind"]
        data[k][(r["precision"], r["spec"]["batch"])].append(
            (size[k](r["spec"]), r["placement"]["frac_ops"].get("ANE", 0.0))
        )
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 3), squeeze=False)
    for ax, (kind, series) in zip(axes[0], sorted(data.items())):
        for (prec, b), pts in sorted(series.items()):
            agg = defaultdict(list)
            for x, y in pts:
                agg[x].append(y)
            xs = sorted(agg)
            ax.plot(
                xs,
                [100 * median(agg[x]) for x in xs],
                marker="o",
                label=f"{prec}, b={b}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel[kind])
        ax.set_ylabel("ops placed on ANE (%)")
        ax.set_title(kind)
        ax.legend(fontsize=7)
    save(fig, "ane_placement.pdf")


def fig_latency(merge_all):
    """Median latency per framework for matmul (baselines)."""
    series = defaultdict(lambda: defaultdict(list))
    for r in rows("baselines", merge_all):
        if (
            r["spec"]["kind"] != "matmul"
            or r["spec"]["batch"] != 1
            or r["precision"] != "fp16"
        ):
            continue
        series[r["framework"]][r["spec"]["m"]].append(r["latency"]["median_ms"])
    if not series:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for fw, d in sorted(series.items()):
        xs = sorted(d)
        ax.plot(xs, [median(d[x]) for x in xs], marker="o", label=fw)
    ax.set_xscale("log", base=2), ax.set_yscale("log")
    ax.set_xlabel("matmul size n (fp16, batch 1)"), ax.set_ylabel("median latency (ms)")
    ax.legend(fontsize=7)
    save(fig, "baseline_latency.pdf")


def fig_efficiency(merge_all):
    pk = peaks(merge_all)
    if not pk:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    series = defaultdict(list)
    for r in rows("peak", merge_all):
        series[(r["backend"], r["precision"])].append((r["size"], r["tflops"]))
    for (be, pr), pts in sorted(series.items()):
        pts.sort()
        ax.plot(
            [p[0] for p in pts],
            [100 * p[1] / pk[(be, pr)] for p in pts],
            marker="o",
            label=f"{be} {pr}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("matmul size n"), ax.set_ylabel("% of measured peak")
    ax.legend(fontsize=6)
    save(fig, "achieved_vs_peak.pdf")


def fig_hybrid_breakdown(merge_all):
    rs = [r for r in rows("hybrid", merge_all) if "forward_coreml" in r]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    labels, parts = [], defaultdict(list)
    for r in rs:
        labels.append(f"w={r['width']}\n{r['units']}")
        parts["Core ML forward"].append(r["forward_coreml"]["median_ms"])
        parts["transfer"].append(r["transfer_to_gpu"]["median_ms"])
        parts["GPU backward+SGD"].append(r["backward_opt_gpu"]["median_ms"])
        parts["weight sync / 100 steps"].append(r["weight_sync_s"] * 1e3 / 100)
    bottom = [0.0] * len(rs)
    for name, vals in parts.items():
        ax.bar(labels, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    pure = [r["pure_mlx_step"]["median_ms"] for r in rs]
    ax.plot(labels, pure, "k_", markersize=18, label="pure MLX step")
    ax.set_ylabel("ms per step"), ax.legend(fontsize=6)
    save(fig, "hybrid_breakdown.pdf")


def fig_convergence(merge_all):
    rs = list(rows("convergence", merge_all))
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for r in rs:
        h = r["history"]
        axes[0].plot(
            [e["epoch"] for e in h], [e["loss"] for e in h], marker="o", label=r["path"]
        )
        axes[1].plot(
            [e["train_s"] for e in h],
            [100 * e["test_acc"] for e in h],
            marker="o",
            label=r["path"],
        )
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("train wall time (s)"), axes[1].set_ylabel("test accuracy (%)")
    axes[0].legend(fontsize=7)
    save(fig, "convergence.pdf")


def fig_train_step(merge_all):
    d = defaultdict(dict)
    for r in rows("train_step", merge_all):
        if r["optimizer"] == "adam":
            d[r["workload"]][r["device"]] = r["latency"]["median_ms"]
    d = {k: v for k, v in d.items() if "cpu" in v and "mlx" in v}
    if not d:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    names = sorted(d)
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], [d[n]["cpu"] for n in names], 0.4, label="NumPy CPU")
    ax.bar([x + 0.2 for x in xs], [d[n]["mlx"] for n in names], 0.4, label="MLX (GPU)")
    ax.set_xticks(list(xs)), ax.set_xticklabels(names, rotation=20, fontsize=7)
    ax.set_ylabel("training step (ms, Adam)"), ax.set_yscale("log"), ax.legend(
        fontsize=7
    )
    save(fig, "train_step.pdf")


def fig_ablation(merge_all):
    d = defaultdict(lambda: defaultdict(float))
    for r in rows("ablation", merge_all):
        d[r["variant"]][r["width"]] = r["latency"]["across_median_ms"]
    if not d:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for v, pts in d.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], marker="o", label=v)
    ax.set_xscale("log", base=2), ax.set_yscale("log")
    ax.set_xlabel("MLP width (3 layers, batch 256)"), ax.set_ylabel("step ms (Adam)")
    ax.legend(fontsize=7)
    save(fig, "ablation.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all", action="store_true", help="merge every run, not just the latest"
    )
    a = ap.parse_args()
    for f in (
        fig_ane_fraction,
        fig_latency,
        fig_efficiency,
        fig_hybrid_breakdown,
        fig_convergence,
        fig_train_step,
    ):
        f(a.all)


if __name__ == "__main__":
    main()
