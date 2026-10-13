"""Core ML conversion, per-op placement (MLComputePlan) and prediction helpers."""

from __future__ import annotations

import os
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, Tuple

COMPUTE_UNITS = ("ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY")
_DEVICE_NAMES = {
    "MLNeuralEngineComputeDevice": "ANE",
    "MLGPUComputeDevice": "GPU",
    "MLCPUComputeDevice": "CPU",
}


def _device_name(dev: Any) -> str:
    return _DEVICE_NAMES.get(type(dev).__name__, type(dev).__name__)


def convert(torch_model, example, precision: str = "fp16", compute_units: str = "ALL"):
    """Trace a torch module and convert it to an ML Program package.

    Returns ``(mlmodel, package_path)``; the package lives in a temp dir that
    the caller owns (``tempfile.TemporaryDirectory`` returned in ``mlmodel._tmp``).
    """
    import coremltools as ct
    import torch

    traced = torch.jit.trace(torch_model, example)
    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=tuple(example.shape))],
        convert_to="mlprogram",
        compute_precision=prec,
        compute_units=getattr(ct.ComputeUnit, compute_units),
        minimum_deployment_target=ct.target.macOS14,
    )
    tmp = tempfile.TemporaryDirectory(prefix="ezbench_")
    path = os.path.join(tmp.name, "model.mlpackage")
    mlmodel.save(path)
    mlmodel._tmp = tmp  # keep the directory alive as long as the model
    return mlmodel, path


def load(path: str, compute_units: str):
    import coremltools as ct

    return ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, compute_units))


def placement(package_path: str, compute_units: str = "ALL") -> Dict[str, Any]:
    """Per-op device placement predicted by Core ML's ``MLComputePlan``.

    ``const`` ops are excluded (no compute). Returns per-op rows and summaries
    by op count and by estimated cost weight.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = ct.utils.compile_model(package_path)
    plan = MLComputePlan.load_from_path(
        compiled, compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    program = plan.model_structure.program
    if program is None:
        raise ValueError("not an ML Program")
    main = program.functions["main"]

    ops, by_count, by_weight = [], Counter(), defaultdict(float)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    for op in main.block.operations:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        preferred = _device_name(usage.preferred_compute_device) if usage else "NONE"
        supported = (
            sorted({_device_name(d) for d in usage.supported_compute_devices})
            if usage
            else []
        )
        weight = float(cost.weight) if cost is not None else 0.0
        ops.append(
            {
                "op": op.operator_name,
                "preferred": preferred,
                "supported": supported,
                "weight": weight,
            }
        )
        by_count[preferred] += 1
        by_weight[preferred] += weight
        by_type[op.operator_name][preferred] += 1
    n = max(sum(by_count.values()), 1)
    return {
        "n_ops": sum(by_count.values()),
        "ops": ops,
        "frac_ops": {d: c / n for d, c in by_count.items()},
        "frac_weight": dict(by_weight),
        "by_op_type": {t: dict(c) for t, c in by_type.items()},
    }


def predict_fn(mlmodel, example_np):
    """Return a zero-arg callable that runs one prediction."""
    feed = {"x": example_np}
    return lambda: mlmodel.predict(feed)
