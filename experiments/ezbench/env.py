"""Capture the hardware / software environment so results are reproducible."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict


def _sh(*cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _version(module: str) -> str:
    from importlib import metadata

    dist = {
        "mlx": "mlx",
        "torch": "torch",
        "coremltools": "coremltools",
        "numpy": "numpy",
        "energizer": "energizer",
    }.get(module, module)
    try:
        return metadata.version(dist)
    except Exception:
        return "unavailable"


def _power_state() -> Dict[str, Any]:
    """AC/battery, charge level and thermal throttling at the time of the run."""
    batt = _sh("pmset", "-g", "batt")
    pct = re.search(r"(\d+)%", batt)
    therm = _sh("pmset", "-g", "therm")
    return {
        "on_ac": "AC Power" in batt.splitlines()[0] if batt else None,
        "battery_pct": int(pct.group(1)) if pct else None,
        "thermal_note": " | ".join(
            ln.strip() for ln in therm.splitlines() if "limit" in ln.lower()
        ),
    }


def capture() -> Dict[str, Any]:
    """Return a JSON-serialisable description of the machine and toolchain."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "model_identifier": _sh("sysctl", "-n", "hw.model"),
        "memory_gb": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1),
        "macos": platform.mac_ver()[0],
        "macos_build": _sh("sw_vers", "-buildVersion"),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "mlx": _version("mlx"),
        "torch": _version("torch"),
        "coremltools": _version("coremltools"),
        "energizer": _version("energizer"),
        "git_commit": _sh("git", "-C", root, "rev-parse", "HEAD"),
        # Results and generated paper outputs are written *by* the run, so they
        # must not count as "dirty" -- only source changes do.
        "git_dirty": bool(
            _sh(
                "git",
                "-C",
                root,
                "status",
                "--porcelain",
                "--",
                ".",
                ":!results",
                ":!paper/tables",
                ":!paper/figures",
            )
        ),
        "power_mode": (
            _sh("pmset", "-g", "batt").splitlines()[0]
            if _sh("pmset", "-g", "batt")
            else ""
        ),
        "low_power_mode": _sh("pmset", "-g").count("lowpowermode         1") > 0,
        **_power_state(),
    }
