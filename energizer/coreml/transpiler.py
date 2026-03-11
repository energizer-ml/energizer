import coremltools as ct
from coremltools.converters.mil import Builder as mb
from .tracer import Tracer, IRNode
from ..tensor import Tensor
import numpy as np


class Transpiler:
    def __init__(self):
        self._op_mapping = {
            "MatMul": self._translate_matmul,
            "Add": self._translate_add,
            "Maximum": self._translate_relu,
            "Transpose": self._translate_transpose,
            "Mul": self._translate_mul,
        }

    def transpile(self, model, example_inputs, output_name="model.mlpackage"):
        """
        Transpile a traced Energizer model into a CoreML package.
        """
        with Tracer() as tracer:
            nodes = tracer.trace(model, example_inputs)

        # Define the input specification based on the example input shape
        input_shape = example_inputs[0].shape
        input_spec = [mb.TensorSpec(shape=input_shape)]

        @mb.program(input_specs=input_spec)
        def prog(x):
            # Maps energizer object ids to CoreML MIL variables
            self.env = {id(example_inputs[0]): x}
            last_out = x

            for node in nodes:
                op_translator = self._op_mapping.get(node.op)
                if not op_translator:
                    raise NotImplementedError(
                        f"Transpilation for op '{node.op}' is not supported yet."
                    )

                last_out = op_translator(node)
                self.env[id(node)] = last_out

            return last_out

        print(f"Converting traced program to CoreML...")
        try:
            coreml_model = ct.convert(prog)
            coreml_model.save(output_name)
            print(f"Model successfully saved to {output_name}")
            return coreml_model
        except Exception as e:
            print(
                f"CoreML Conversion Failed. Note: This could be due to Python 3.14 incompatibility with coremltools."
            )
            raise e

    def _get_input_var(self, inp):
        if isinstance(inp, IRNode):
            return self.env[id(inp)]
        elif isinstance(inp, Tensor):
            return mb.const(val=inp.numpy())
        else:
            return mb.const(val=inp)

    def _translate_matmul(self, node):
        x = self._get_input_var(node.inputs[0])
        w = self._get_input_var(node.inputs[1])
        return mb.matmul(x=x, y=w)

    def _translate_add(self, node):
        a = self._get_input_var(node.inputs[0])
        b = self._get_input_var(node.inputs[1])
        return mb.add(x=a, y=b)

    def _translate_relu(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.relu(x=x)

    def _translate_transpose(self, node):
        x = self._get_input_var(node.inputs[0])
        # Find dimension logic based on shape
        inp = node.inputs[0]
        rank = len(inp.shape) if hasattr(inp, 'shape') else 1 # Fallback to 1 if scalar/unknown
        perm = list(reversed(range(max(1, rank))))
        return mb.transpose(x=x, perm=perm)

    def _translate_mul(self, node):
        x = self._get_input_var(node.inputs[0])
        y = self._get_input_var(node.inputs[1])
        return mb.multiply(x=x, y=y)


def compile_to_coreml(model, example_input, output_path="model.mlpackage"):
    transpiler = Transpiler()
    if not isinstance(example_input, tuple):
        example_input = (example_input,)
    return transpiler.transpile(model, example_input, output_path)
