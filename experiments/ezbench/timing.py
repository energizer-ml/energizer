"""Latency measurement with explicit warm-up and robust statistics."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List


def measure(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    min_seconds: float = 0.0,
) -> Dict[str, float]:
    """Time ``fn`` (which must block until the result is materialised).

    Runs ``warmup`` untimed calls, then at least ``iters`` timed calls and at
    least ``min_seconds`` of wall-clock time. Returns milliseconds.
    """
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    start = time.perf_counter()
    while len(samples) < iters or (time.perf_counter() - start) < min_seconds:
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    n = len(samples)
    return {
        "n": n,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "p05_ms": samples[int(0.05 * (n - 1))],
        "p95_ms": samples[int(0.95 * (n - 1))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def repeat(
    make_fn: Callable[[], Callable[[], object]], *, repeats: int = 5, **kw
) -> Dict[str, float]:
    """Run :func:`measure` ``repeats`` times on a fresh callable each time.

    Separates within-run noise (``std_ms`` of one run) from run-to-run variation
    (``across_*``), which is what thermal drift and scheduler state produce.
    """
    runs = [measure(make_fn(), **kw) for _ in range(repeats)]
    meds = [r["median_ms"] for r in runs]
    out = dict(runs[len(runs) // 2])  # representative run
    out.update(
        repeats=repeats,
        across_median_ms=statistics.median(meds),
        across_mean_ms=statistics.fmean(meds),
        across_std_ms=statistics.pstdev(meds),
        across_min_ms=min(meds),
        across_max_ms=max(meds),
        medians_ms=meds,
    )
    return out
