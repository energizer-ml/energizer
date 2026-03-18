import coremltools as ct
from coremltools.converters.mil import Builder as mb
from .bouncer import Bouncer, BouncerConfig
from .tracer import Tracer, IRNode
from energizer.tensor import Tensor
import numpy as np


class Transpiler:
    def __init__(
        self, optimize_for_ane: bool = True, bouncer_config: BouncerConfig | None = None
    ):
        self.optimize_for_ane = optimize_for_ane
        self.bouncer = Bouncer(bouncer_config) if optimize_for_ane else None
        self._op_mapping = {
            "MatMul": self._translate_matmul,
            "Add": self._translate_add,
            "Linear": self._translate_linear,
            "LayerNorm": self._translate_layer_norm,
            "Maximum": self._translate_relu,
            "Transpose": self._translate_transpose,
            "Mul": self._translate_mul,
            "Sigmoid": self._translate_sigmoid,
            "Softmax": self._translate_softmax,
            "GELU": self._translate_gelu,
            "ReLU": self._translate_relu,
            "SiLU": self._translate_silu,
            "ScaledDotProductAttention": self._translate_scaled_dot_product_attention,
        }
        self.last_traced_nodes = []
        self.last_bounced_nodes = []

    def transpile(self, model, example_inputs, output_name="model.mlpackage"):
        """
        Transpile a traced Energizer model into a CoreML package.
        """
        with Tracer() as tracer:
            nodes = tracer.trace(model, example_inputs)
        self.last_traced_nodes = nodes
        if self.bouncer is not None:
            nodes = self.bouncer.bounce(nodes)
        self.last_bounced_nodes = nodes
        return self.transpile_nodes(nodes, example_inputs, output_name)

    def transpile_nodes(self, nodes, example_inputs, output_name="model.mlpackage"):
        self.last_bounced_nodes = nodes

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
            convert_kwargs = {}
            if self.optimize_for_ane and hasattr(ct, "precision"):
                convert_kwargs["compute_precision"] = ct.precision.FLOAT16
            if any(node.op == "ScaledDotProductAttention" for node in nodes):
                target = getattr(getattr(ct, "target", None), "macOS15", None)
                if target is not None:
                    convert_kwargs["minimum_deployment_target"] = target
            coreml_model = ct.convert(prog, **convert_kwargs)
            coreml_model.save(output_name)
            print(f"Model successfully saved to {output_name}")
            return coreml_model
        except Exception as e:
            print(
                f"CoreML Conversion Failed. Note: This could be due to Python 3.14 incompatibility with coremltools."
            )
            raise e

    def _get_input_var(self, inp):
        if id(inp) in getattr(self, "env", {}):
            return self.env[id(inp)]
        if isinstance(inp, IRNode):
            return self.env[id(inp)]
        elif isinstance(inp, Tensor):
            return mb.const(val=inp.numpy())
        else:
            return mb.const(val=inp)

    def _const_value(self, inp):
        if isinstance(inp, Tensor):
            return inp.numpy()
        if isinstance(inp, np.ndarray):
            return inp
        if hasattr(inp, "data"):
            data = inp.data
            if hasattr(data, "tolist") and not isinstance(data, np.ndarray):
                return np.array(data.tolist(), dtype=np.float32)
            return np.array(data, dtype=np.float32)
        if np.isscalar(inp):
            return np.array(inp, dtype=np.float32)
        return np.array(inp, dtype=np.float32)

    def _translate_matmul(self, node):
        x = self._get_input_var(node.inputs[0])
        w = self._get_input_var(node.inputs[1])
        return mb.matmul(x=x, y=w)

    def _translate_add(self, node):
        a = self._get_input_var(node.inputs[0])
        b = self._get_input_var(node.inputs[1])
        return mb.add(x=a, y=b)

    def _translate_linear(self, node):
        x = self._get_input_var(node.inputs[0])
        weight = self._const_value(node.inputs[1])
        bias = self._const_value(node.inputs[2])
        return mb.linear(
            x=x,
            weight=weight.T.astype(np.float32, copy=False),
            bias=bias.astype(np.float32, copy=False),
        )

    def _translate_layer_norm(self, node):
        x = self._get_input_var(node.inputs[0])
        axes = self._const_value(node.inputs[1]).astype(np.int32, copy=False)
        gamma = self._const_value(node.inputs[2]).astype(np.float32, copy=False)
        beta = self._const_value(node.inputs[3]).astype(np.float32, copy=False)
        epsilon = float(self._const_value(node.inputs[4]).reshape(()))
        return mb.layer_norm(
            x=x,
            axes=axes,
            gamma=gamma,
            beta=beta,
            epsilon=epsilon,
        )

    def _translate_relu(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.relu(x=x)

    def _translate_transpose(self, node):
        x = self._get_input_var(node.inputs[0])
        # Find dimension logic based on shape
        inp = node.inputs[0]
        rank = (
            len(inp.shape) if hasattr(inp, "shape") else 1
        )  # Fallback to 1 if scalar/unknown
        perm = list(reversed(range(max(1, rank))))
        return mb.transpose(x=x, perm=perm)

    def _translate_mul(self, node):
        x = self._get_input_var(node.inputs[0])
        y = self._get_input_var(node.inputs[1])
        return mb.multiply(x=x, y=y)

    def _translate_sigmoid(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.sigmoid(x=x)

    def _translate_softmax(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.softmax(x=x, axis=-1)

    def _translate_gelu(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.gelu(x=x, mode="TANH_APPROXIMATION")

    def _translate_silu(self, node):
        x = self._get_input_var(node.inputs[0])
        return mb.silu(x=x)

    def _translate_scaled_dot_product_attention(self, node):
        query = self._get_input_var(node.inputs[0])
        key = self._get_input_var(node.inputs[1])
        value = self._get_input_var(node.inputs[2])
        scale = self._const_value(node.inputs[3]).astype(np.float32, copy=False)
        attn_mask = None
        if len(node.inputs) > 4 and node.inputs[4] is not None:
            attn_mask = self._get_input_var(node.inputs[4])

        if hasattr(mb, "scaled_dot_product_attention"):
            scaled_query = mb.mul(x=query, y=scale)
            kwargs = dict(
                query=scaled_query,
                key=key,
                value=value,
            )
            if attn_mask is not None:
                kwargs["attn_mask"] = attn_mask
            return mb.scaled_dot_product_attention(**kwargs)

        scores = mb.matmul(x=query, y=mb.transpose(x=key, perm=[0, 2, 1]))
        if float(scale.reshape(())) != 1.0:
            scores = mb.mul(x=scores, y=scale)
        if attn_mask is not None:
            scores = mb.add(x=scores, y=attn_mask)
        weights = mb.softmax(x=scores, axis=-1)
        return mb.matmul(x=weights, y=value)


def compile_to_coreml(
    model,
    example_input,
    output_path="model.mlpackage",
    optimize_for_ane: bool = True,
    bouncer_config: BouncerConfig | None = None,
):
    transpiler = Transpiler(
        optimize_for_ane=optimize_for_ane, bouncer_config=bouncer_config
    )
    if not isinstance(example_input, tuple):
        example_input = (example_input,)
    return transpiler.transpile(model, example_input, output_path)
