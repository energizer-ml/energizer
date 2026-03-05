from __future__ import annotations
from typing import Any
import numpy as np
import mlx.core as mx


def _get_data_ref(x: Any):
    return x.data if hasattr(x, "data") else x


def zeros_(tensor: Any) -> Any:
    data = _get_data_ref(tensor)
    if mx is not None and isinstance(data, mx.array):
        z = mx.zeros_like(data)
    else:
        z = np.zeros_like(np.array(data))
    if hasattr(tensor, "data"):
        tensor.data = z
        return tensor
    return z


def orthogonal_(tensor: Any, gain: float = 1.0, seed: int | None = None) -> Any:
    data = _get_data_ref(tensor)
    shape = getattr(data, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("orthogonal_ expects a tensor with at least 2 dimensions")
    if seed is not None:
        rng = np.random.default_rng(seed)
        randn = rng.standard_normal
    else:
        randn = np.random.randn
    rows = int(shape[0])
    cols = int(np.prod(shape[1:]))
    flat_shape = (rows, cols)
    a = randn(*flat_shape).astype(np.float32)
    transposed = False
    if rows < cols:
        a = a.T
        transposed = True
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    q *= np.sign(d + 1e-12)
    if transposed:
        q = q.T
    q = (q.reshape(shape) * float(gain)).astype(np.float32)
    if mx is not None and isinstance(data, mx.array):
        out = mx.array(q)
    else:
        out = q
    if hasattr(tensor, "data"):
        tensor.data = out
        return tensor
    return out
