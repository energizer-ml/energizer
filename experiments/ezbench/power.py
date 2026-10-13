"""Power / energy sampling through ``powermetrics`` (needs root).

Usage::

    with PowerSampler(interval_ms=100) as ps:
        run_workload()
    print(ps.summary())   # mean mW per rail + integrated energy in mJ

If ``powermetrics`` cannot be started (not root, not macOS) the sampler is
inert and ``summary()`` returns ``None`` -- numbers are never fabricated.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional

_PAT = {
    "cpu": re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.I),
    "gpu": re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.I),
    "ane": re.compile(r"^ANE Power:\s*([\d.]+)\s*mW", re.I),
    "combined": re.compile(r"^Combined Power.*?:\s*([\d.]+)\s*mW", re.I),
}


class PowerSampler:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: List[Dict[str, float]] = []
        self.active = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._t1 = 0.0

    @staticmethod
    def available() -> bool:
        return os.geteuid() == 0 and os.path.exists("/usr/bin/powermetrics")

    def __enter__(self) -> "PowerSampler":
        if not self.available():
            return self
        self._proc = subprocess.Popen(
            [
                "/usr/bin/powermetrics",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
                "-i",
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.active = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        cur: Dict[str, float] = {}
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if line.startswith("*** Sampled system activity"):
                if cur:
                    self.samples.append(cur)
                cur = {}
                continue
            for key, pat in _PAT.items():
                m = pat.match(line)
                if m:
                    cur[key] = float(m.group(1))
        if cur:
            self.samples.append(cur)

    def __exit__(self, *exc) -> None:
        self._t1 = time.perf_counter()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> Optional[Dict[str, float]]:
        if not self.active or not self.samples:
            return None
        # The first sample covers time before the workload started.
        samples = self.samples[1:] or self.samples
        out: Dict[str, float] = {
            "n_samples": len(samples),
            "interval_ms": self.interval_ms,
        }
        for rail in ("cpu", "gpu", "ane", "combined"):
            vals = [s[rail] for s in samples if rail in s]
            if vals:
                mean = sum(vals) / len(vals)
                out[f"{rail}_mw_mean"] = mean
                out[f"{rail}_mj"] = (
                    mean * len(vals) * self.interval_ms / 1e3
                )  # mW*s = mJ
        return out


def idle_baseline(seconds: float = 10.0, interval_ms: int = 100):
    """Mean power (mW per rail) of the idle machine; subtract from workload power."""
    if not PowerSampler.available():
        return None
    with PowerSampler(interval_ms) as ps:
        time.sleep(seconds)
    return ps.summary()


def net(summary, idle, duration_s: float):
    """Workload power/energy with the idle baseline removed (mW, mJ)."""
    if not summary or not idle:
        return None
    out = {}
    for rail in ("cpu", "gpu", "ane", "combined"):
        k = f"{rail}_mw_mean"
        if k in summary and k in idle:
            mw = max(summary[k] - idle[k], 0.0)
            out[f"{rail}_net_mw"] = mw
            out[f"{rail}_net_mj"] = mw * duration_s
    return out
