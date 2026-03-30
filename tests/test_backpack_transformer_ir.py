import os

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")

import numpy as np

from energizer import Tensor
from energizer.backpack.compiler.tracer import IRNode
from energizer.backpack.compiler.transpiler import Transpiler


def test_transpiler_compiles_layer_norm_ir_node(tmp_path):
    x = Tensor.randn(2, 8)
    gamma = np.ones((8,), dtype=np.float32)
    beta = np.zeros((8,), dtype=np.float32)
    axes = np.array([-1], dtype=np.int32)
    epsilon = np.array(1e-5, dtype=np.float32)

    layer_norm = IRNode(
        "LayerNorm",
        [x, axes, gamma, beta, epsilon],
        (2, 8),
        "float32",
        {"ane_preferred": True},
    )

    package_path = tmp_path / "layer_norm_ir.mlpackage"
    transpiler = Transpiler(optimize_for_ane=True)
    transpiler.transpile_nodes([layer_norm], (x,), str(package_path))

    assert [node.op for node in transpiler.last_bounced_nodes] == ["LayerNorm"]
    assert package_path.exists()
