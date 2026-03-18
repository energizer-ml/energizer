import os
import platform
import shutil
from pathlib import Path

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")

import numpy as np
import pytest

from energizer import Tensor
from energizer.backpack.compiler.transpiler import Transpiler
from energizer.backpack.runner import Runner


class ShortMLP:
    def __init__(self, input_size=16, hidden_size=8, output_size=4):
        self.w1 = Tensor.randn(input_size, hidden_size) * 0.1
        self.b1 = Tensor.zeros(hidden_size)
        self.w2 = Tensor.randn(hidden_size, output_size) * 0.1
        self.b2 = Tensor.zeros(output_size)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        hidden = (x @ self.w1) + self.b1
        hidden = hidden.maximum(0.0)
        return (hidden @ self.w2) + self.b2


def test_transpiler_bounces_short_mlp_to_ane_friendly_ops(tmp_path):
    model = ShortMLP()
    example = Tensor.randn(1, 16)
    package_path = tmp_path / "short_mlp.mlpackage"

    transpiler = Transpiler(optimize_for_ane=True)
    transpiler.transpile(model, (example,), str(package_path))

    bounced_ops = [node.op for node in transpiler.last_bounced_nodes]
    assert bounced_ops == ["Linear", "ReLU", "Linear"]
    assert all(
        node.attrs.get("ane_preferred") for node in transpiler.last_bounced_nodes
    )
    assert package_path.exists()


@pytest.mark.skipif(platform.system() != "Darwin", reason="CoreML runtime requires macOS")
def test_runner_uses_ane_oriented_runtime_settings_for_short_mlp():
    model = ShortMLP()
    example = Tensor.randn(1, 16)
    artifacts_dir = Path.cwd() / ".pytest_coreml_artifacts"
    package_path = artifacts_dir / "short_mlp_runtime.mlpackage"
    if package_path.exists():
        shutil.rmtree(package_path)
    artifacts_dir.mkdir(exist_ok=True)

    try:
        transpiler = Transpiler(optimize_for_ane=True)
        transpiler.transpile(model, (example,), str(package_path))

        runner = Runner.load(
            str(package_path), compute_units="ALL", fp16_inputs=True, warmup_runs=0
        )
        if getattr(runner._model, "_framework_error", None) is not None:
            pytest.skip(f"CoreML runtime unavailable: {runner._model._framework_error}")

        try:
            out = runner(np.random.randn(*runner.input_shape).astype(np.float32))
        except RuntimeError as exc:
            pytest.skip(f"CoreML runtime unavailable: {exc}")

        assert runner._config.compute_units == "ALL"
        assert runner._config.fp16_inputs is True
        assert out.shape == (1, 4)
    finally:
        if package_path.exists():
            shutil.rmtree(package_path)
