"""Append-only JSON Lines result files, one per script run."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from . import env as _env

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)


class ResultWriter:
    def __init__(self, experiment: str, out_dir: str | None = None):
        self.experiment = experiment
        out_dir = out_dir or os.path.join(RESULTS_DIR, experiment)
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.path = os.path.join(out_dir, f"{stamp}.jsonl")
        self._env = _env.capture()
        with open(self.path, "w") as f:
            f.write(
                json.dumps({"type": "env", "experiment": experiment, **self._env})
                + "\n"
            )

    def write(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps({"type": "result", "experiment": self.experiment, **record})
                + "\n"
            )


def load(path: str):
    env, rows = None, []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "env":
                env = obj
            else:
                rows.append(obj)
    return env, rows
