"""Small, parametrised benchmark models.

The same architecture is defined for PyTorch (source of the Core ML conversion
and the MPS baseline) and for MLX (MLX baseline), so every framework runs an
identical computation. Weights are random: only speed / placement matter here.

Workload specs are plain dicts so they serialise straight into result files::

    {"kind": "conv",        "batch": 1, "channels": 64, "hw": 64, "kernel": 3}
    {"kind": "matmul",      "batch": 1, "m": 512, "k": 512, "n": 512}
    {"kind": "transformer", "batch": 1, "seq": 128, "d_model": 256, "heads": 4, "ffn_mult": 4}
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

Spec = Dict[str, Any]


def name(spec: Spec) -> str:
    k = spec["kind"]
    if k == "conv":
        return f"conv_c{spec['channels']}_hw{spec['hw']}_k{spec['kernel']}_b{spec['batch']}"
    if k == "matmul":
        return f"matmul_{spec['m']}x{spec['k']}x{spec['n']}_b{spec['batch']}"
    if k == "transformer":
        return (
            f"tfm_s{spec['seq']}_d{spec['d_model']}_h{spec['heads']}_b{spec['batch']}"
        )
    raise ValueError(k)


def input_shape(spec: Spec, layout: str = "nchw") -> Tuple[int, ...]:
    k, b = spec["kind"], spec["batch"]
    if k == "conv":
        c, hw = spec["channels"], spec["hw"]
        return (b, c, hw, hw) if layout == "nchw" else (b, hw, hw, c)
    if k == "matmul":
        return (b, spec["m"], spec["k"])
    if k == "transformer":
        return (b, spec["seq"], spec["d_model"])
    raise ValueError(k)


def flops(spec: Spec) -> float:
    """Multiply-accumulate count * 2 for the dominant ops (for TFLOP/s plots)."""
    k, b = spec["kind"], spec["batch"]
    if k == "conv":
        c, hw, ks = spec["channels"], spec["hw"], spec["kernel"]
        return 2.0 * b * hw * hw * c * c * ks * ks
    if k == "matmul":
        return 2.0 * b * spec["m"] * spec["k"] * spec["n"]
    if k == "transformer":
        s, d, f = spec["seq"], spec["d_model"], spec["ffn_mult"]
        proj = 2.0 * s * d * d * 4  # qkv + out
        attn = 2.0 * 2 * s * s * d  # QK^T and AV
        ffn = 2.0 * 2 * s * d * d * f
        return b * (proj + attn + ffn)
    raise ValueError(k)


# ── PyTorch ───────────────────────────────────────────────────────────────────


def build_torch(spec: Spec):
    """Return ``(module, example_input)`` on CPU, in eval mode."""
    import torch
    import torch.nn as nn

    k = spec["kind"]
    if k == "conv":
        ks = spec["kernel"]
        model = nn.Conv2d(spec["channels"], spec["channels"], ks, padding=ks // 2)
    elif k == "matmul":
        model = nn.Linear(spec["k"], spec["n"], bias=False)
    elif k == "transformer":

        class Block(nn.Module):
            # Shapes are python ints fixed at construction: tracing through
            # ``x.shape`` produces int ops that coremltools' torch frontend
            # rejects on recent torch versions.
            def __init__(self, s: int, d: int, h: int, f: int):
                super().__init__()
                self.s, self.h, self.d, self.dh = s, h, d, d // h
                self.scale = float(self.dh) ** -0.5
                self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
                self.qkv = nn.Linear(d, 3 * d)
                self.proj = nn.Linear(d, d)
                self.fc1, self.fc2 = nn.Linear(d, f * d), nn.Linear(f * d, d)
                self.act = nn.GELU()

            def forward(self, x):
                q, k_, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
                q, k_, v = (
                    t.reshape(-1, self.s, self.h, self.dh).transpose(1, 2)
                    for t in (q, k_, v)
                )
                att = torch.softmax((q @ k_.transpose(-1, -2)) * self.scale, dim=-1)
                y = (att @ v).transpose(1, 2).reshape(-1, self.s, self.d)
                x = x + self.proj(y)
                return x + self.fc2(self.act(self.fc1(self.ln2(x))))

        model = Block(spec["seq"], spec["d_model"], spec["heads"], spec["ffn_mult"])
    else:
        raise ValueError(k)
    model.eval()
    return model, torch.randn(*input_shape(spec))


# ── MLX ───────────────────────────────────────────────────────────────────────


def build_mlx(spec: Spec):
    """Return ``(callable, example_input)`` using MLX (NHWC for convolutions)."""
    import mlx.core as mx
    import mlx.nn as nn

    k = spec["kind"]
    if k == "conv":
        ks = spec["kernel"]
        layer = nn.Conv2d(spec["channels"], spec["channels"], ks, padding=ks // 2)
        x = mx.random.normal(input_shape(spec, layout="nhwc"))
        fn = layer
    elif k == "matmul":
        layer = nn.Linear(spec["k"], spec["n"], bias=False)
        x = mx.random.normal(input_shape(spec))
        fn = layer
    elif k == "transformer":
        d, h, f = spec["d_model"], spec["heads"], spec["ffn_mult"]

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
                self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
                self.fc1, self.fc2 = nn.Linear(d, f * d), nn.Linear(f * d, d)

            def __call__(self, x):
                b, s, _ = x.shape
                q, k_, v = mx.split(self.qkv(self.ln1(x)), 3, axis=-1)
                q, k_, v = (
                    t.reshape(b, s, h, d // h).transpose(0, 2, 1, 3) for t in (q, k_, v)
                )
                att = mx.softmax(
                    (q @ k_.transpose(0, 1, 3, 2)) / (d // h) ** 0.5, axis=-1
                )
                y = (att @ v).transpose(0, 2, 1, 3).reshape(b, s, d)
                x = x + self.proj(y)
                return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))

        fn = Block()
        x = mx.random.normal(input_shape(spec))
    else:
        raise ValueError(k)
    mx.eval(fn.parameters())
    return fn, x
