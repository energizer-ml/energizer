"""
backpack.runner
───────────────
CoreML inference engine for Energizer models.

Workflow:
    1. Train your model normally with Energizer (MLX/NumPy backend)
    2. Compile it once:  backpack.compile(model, example_input)  → .mlpackage
    3. Run it fast:      runner = backpack.runner.load("model.mlpackage")
                         output = runner(input_tensor)

The runner handles:
    - float32 ↔ float16 casting (ANE requires fp16)
    - numpy ↔ Energizer Tensor conversion
    - warm-up pass on load (avoids first-call JIT latency)
    - basic profiling (latency, throughput)
    - device visibility (which compute units were used)
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """Configuration for the CoreML runner."""

    # Which compute units CoreML is allowed to use.
    # "ALL"        → CoreML decides (ANE + GPU + CPU), best for ANE dispatch
    # "CPU_AND_GPU"→ skips ANE (useful for debugging)
    # "CPU_ONLY"   → forces CPU only
    compute_units: str = "ALL"

    # Cast inputs to float16 before inference (required for ANE dispatch).
    # Set to False only if your model was compiled with float32 precision.
    fp16_inputs: bool = True

    # Number of warm-up passes on load (burns JIT compilation latency).
    warmup_runs: int = 2

    # If True, keep a rolling window of per-call latencies.
    track_latency: bool = True
    latency_window: int = 100


@dataclass
class RunStats:
    """Accumulated statistics for a runner session."""
    calls:           int   = 0
    total_ms:        float = 0.0
    latencies_ms:    list  = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def last_ms(self) -> float:
        return self.latencies_ms[-1] if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.99)]

    def record(self, ms: float, window: int):
        self.calls      += 1
        self.total_ms   += ms
        self.latencies_ms.append(ms)
        if len(self.latencies_ms) > window:
            self.latencies_ms.pop(0)

    def __repr__(self) -> str:
        return (
            f"RunStats(calls={self.calls}, "
            f"avg={self.avg_ms:.2f}ms, "
            f"last={self.last_ms:.2f}ms, "
            f"p99={self.p99_ms:.2f}ms)"
        )


# ── Runner ────────────────────────────────────────────────────────────────────

class Runner:
    """
    Wraps a compiled CoreML .mlpackage for fast inference.

    Usage:
        runner = Runner.load("my_model.mlpackage")
        output = runner(input_tensor)           # Energizer Tensor in/out
        output = runner.predict_numpy(arr)      # numpy in/out (lower overhead)
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[RunnerConfig] = None,
    ):
        self._path   = model_path
        self._config = config or RunnerConfig()
        self._model  = None
        self._input_name: str  = ""
        self._output_name: str = ""
        self._input_shape: tuple = ()
        self.stats = RunStats()

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools is required to use backpack.runner.\n"
                "Install it with:  pip install coremltools\n"
                "Note: requires Python ≤ 3.12."
            )

        # Map config string to CoreML compute units enum
        cu_map = {
            "ALL":         ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        }
        cu = cu_map.get(self._config.compute_units.upper(), ct.ComputeUnit.ALL)

        self._model = ct.models.MLModel(self._path, compute_units=cu)

        spec = self._model.get_spec()
        self._input_name  = spec.description.input[0].name
        self._output_name = spec.description.output[0].name

        # Infer input shape from spec
        feat = spec.description.input[0]
        if feat.type.HasField("multiArrayType"):
            self._input_shape = tuple(feat.type.multiArrayType.shape)

        # Warm-up: run dummy passes to burn CoreML's JIT compilation
        if self._config.warmup_runs > 0 and self._input_shape:
            dummy = np.zeros(self._input_shape, dtype=np.float16
                             if self._config.fp16_inputs else np.float32)
            for _ in range(self._config.warmup_runs):
                self._model.predict({self._input_name: dummy})

    # ── Core predict ──────────────────────────────────────────────────────────

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on a numpy array.
        Input is cast to fp16 if config.fp16_inputs is True.
        Output is always returned as float32 numpy array.
        """
        if self._config.fp16_inputs:
            x = x.astype(np.float16)
        else:
            x = x.astype(np.float32)

        t0     = time.perf_counter()
        result = self._model.predict({self._input_name: x})
        ms     = (time.perf_counter() - t0) * 1000.0

        if self._config.track_latency:
            self.stats.record(ms, self._config.latency_window)

        out = result[self._output_name]
        return np.asarray(out, dtype=np.float32)

    def __call__(self, x) -> "Tensor":
        """
        Run inference on an Energizer Tensor.
        Returns an Energizer Tensor on CPU.
        """
        from energizer.tensor import Tensor

        if isinstance(x, Tensor):
            arr = x.numpy()
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            arr = np.asarray(x, dtype=np.float32)

        out_arr = self.predict_numpy(arr)

        return Tensor(out_arr, device="cpu", requires_grad=False)

    # ── Batch inference ───────────────────────────────────────────────────────

    def predict_batch(self, inputs: list) -> list:
        """
        Run inference on a list of inputs (Tensors or numpy arrays).
        Returns a list of Energizer Tensors.
        Useful when CoreML model expects batch_size=1 (ANE constraint).
        """
        return [self(x) for x in inputs]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def benchmark(self, x, runs: int = 50) -> dict:
        """
        Run `runs` inference passes and return latency statistics.

        Returns:
            dict with keys: runs, avg_ms, min_ms, max_ms, p50_ms, p99_ms
        """
        if isinstance(x, type) and hasattr(x, "numpy"):
            arr = x.numpy()
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            arr = np.asarray(x, dtype=np.float32)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict_numpy(arr)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies.sort()
        n = len(latencies)
        return {
            "runs":   n,
            "avg_ms": sum(latencies) / n,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "p50_ms": latencies[n // 2],
            "p99_ms": latencies[int(n * 0.99)],
        }

    def describe(self) -> str:
        """Human-readable summary of the loaded model."""
        lines = [
            f"Runner  →  {self._path}",
            f"  input   : {self._input_name}  shape={self._input_shape}",
            f"  output  : {self._output_name}",
            f"  compute : {self._config.compute_units}",
            f"  fp16    : {self._config.fp16_inputs}",
            f"  warmup  : {self._config.warmup_runs} runs",
            f"  stats   : {self.stats}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Runner(path='{self._path}', compute='{self._config.compute_units}')"

    # ── Class-level factory ───────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str,
        compute_units: str = "ALL",
        fp16_inputs: bool = True,
        warmup_runs: int = 2,
    ) -> "Runner":
        """
        Convenience factory.

        Example:
            runner = Runner.load("model.mlpackage")
            output = runner(my_tensor)
        """
        config = RunnerConfig(
            compute_units=compute_units,
            fp16_inputs=fp16_inputs,
            warmup_runs=warmup_runs,
        )
        return cls(path, config)


# ── Module-level convenience ──────────────────────────────────────────────────

def load(
    path: str,
    compute_units: str = "ALL",
    fp16_inputs: bool = True,
    warmup_runs: int = 2,
) -> Runner:
    """
    Load a compiled .mlpackage and return a ready-to-use Runner.

    Args:
        path:          Path to the .mlpackage directory.
        compute_units: "ALL" (default) | "CPU_AND_GPU" | "CPU_ONLY"
        fp16_inputs:   Cast inputs to float16 (required for ANE, default True).
        warmup_runs:   Number of warm-up inference passes on load (default 2).

    Returns:
        Runner instance.

    Example:
        import backpack.runner as runner

        model  = runner.load("my_model.mlpackage")
        output = model(input_tensor)
        print(model.stats)
    """
    return Runner.load(path, compute_units=compute_units,
                       fp16_inputs=fp16_inputs, warmup_runs=warmup_runs)


def benchmark(
    path: str,
    example_input: np.ndarray,
    runs: int = 50,
    compute_units: str = "ALL",
) -> dict:
    """
    One-shot benchmark of a compiled model.

    Example:
        import numpy as np
        import backpack.runner as runner

        stats = runner.benchmark("model.mlpackage", np.zeros((1, 784)))
        print(f"avg latency: {stats['avg_ms']:.2f} ms")
    """
    r = load(path, compute_units=compute_units)
    return r.benchmark(example_input, runs=runs)