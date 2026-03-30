import os

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")

import energizer
from energizer import Tensor
from energizer.backpack.compiler.transpiler import Transpiler


class ShortGeluMLP:
    def __init__(self, input_size=16, hidden_size=8, output_size=4):
        self.w1 = Tensor.randn(input_size, hidden_size) * 0.1
        self.b1 = Tensor.zeros(hidden_size)
        self.w2 = Tensor.randn(hidden_size, output_size) * 0.1
        self.b2 = Tensor.zeros(output_size)

    def __call__(self, x):
        hidden = (x @ self.w1) + self.b1
        hidden = energizer.GELU()(hidden)
        return (hidden @ self.w2) + self.b2


def test_transpiler_keeps_gelu_as_first_class_coreml_op(tmp_path):
    model = ShortGeluMLP()
    example = Tensor.randn(1, 16)
    package_path = tmp_path / "short_gelu_mlp.mlpackage"

    transpiler = Transpiler(optimize_for_ane=True)
    transpiler.transpile(model, (example,), str(package_path))

    bounced_ops = [node.op for node in transpiler.last_bounced_nodes]
    assert bounced_ops == ["Linear", "GELU", "Linear"]
    assert package_path.exists()
