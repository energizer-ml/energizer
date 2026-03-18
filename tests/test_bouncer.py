import os

os.environ.setdefault("ENERGIZER_DISABLE_MLX", "1")

import numpy as np

from energizer.backpack.compiler.bouncer import Bouncer
from energizer.backpack.compiler.tracer import IRNode


def test_bouncer_fuses_matmul_add_into_linear():
    x = object()
    weight = np.ones((8, 4), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)

    matmul = IRNode("MatMul", [x, weight], (2, 4), "float32")
    add = IRNode("Add", [matmul, bias], (2, 4), "float32")

    bounced = Bouncer().bounce([matmul, add])

    assert len(bounced) == 1
    assert bounced[0].op == "Linear"
    assert bounced[0].inputs == [x, weight, bias]
    assert bounced[0].attrs["ane_preferred"] is True


def test_bouncer_canonicalizes_maximum_zero_into_relu():
    x = object()
    zero = np.array(0.0, dtype=np.float32)
    node = IRNode("Maximum", [x, zero], (2, 4), "float32")

    bounced = Bouncer().bounce([node])

    assert len(bounced) == 1
    assert bounced[0].op == "ReLU"
    assert bounced[0].inputs == [x]


def test_bouncer_fuses_sigmoid_mul_into_silu():
    x = object()
    sigmoid = IRNode("Sigmoid", [x], (2, 4), "float32")
    mul = IRNode("Mul", [sigmoid, x], (2, 4), "float32")

    bounced = Bouncer().bounce([sigmoid, mul])

    assert len(bounced) == 1
    assert bounced[0].op == "SiLU"
    assert bounced[0].inputs == [x]


def test_bouncer_fuses_attention_pattern_and_drops_dead_nodes():
    q = object()
    k = object()
    v = object()
    kt = IRNode("Transpose", [k], (1, 8, 8), "float32")
    scores = IRNode("MatMul", [q, kt], (1, 8, 8), "float32")
    scaled = IRNode("Mul", [scores, np.array(0.5, dtype=np.float32)], (1, 8, 8), "float32")
    weights = IRNode("Softmax", [scaled], (1, 8, 8), "float32")
    out = IRNode("MatMul", [weights, v], (1, 8, 16), "float32")

    bounced = Bouncer().bounce([kt, scores, scaled, weights, out])

    assert len(bounced) == 1
    assert bounced[0].op == "ScaledDotProductAttention"
    assert bounced[0].inputs[:3] == [q, k, v]
    assert float(bounced[0].inputs[3]) == 0.5


def test_bouncer_fuses_masked_attention_pattern():
    q = object()
    k = object()
    v = object()
    mask = object()
    kt = IRNode("Transpose", [k], (1, 8, 8), "float32")
    scores = IRNode("MatMul", [q, kt], (1, 8, 8), "float32")
    scaled = IRNode("Mul", [scores, np.array(0.125, dtype=np.float32)], (1, 8, 8), "float32")
    masked = IRNode("Add", [scaled, mask], (1, 8, 8), "float32")
    weights = IRNode("Softmax", [masked], (1, 8, 8), "float32")
    out = IRNode("MatMul", [weights, v], (1, 8, 16), "float32")

    bounced = Bouncer().bounce([kt, scores, scaled, masked, weights, out])

    assert len(bounced) == 1
    assert bounced[0].op == "ScaledDotProductAttention"
    assert bounced[0].inputs[:3] == [q, k, v]
    assert float(bounced[0].inputs[3]) == 0.125
    assert bounced[0].inputs[4] is mask
