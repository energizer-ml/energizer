"""
Backend dispatcher for Energizer.

Backends (device strings):
    - "cpu"  → NumPy, runs on the CPU
    - "mlx"  → Apple MLX, runs on the Apple-silicon GPU through Metal

MLX does **not** execute on the Apple Neural Engine (ANE). The ANE is only
reachable through Core ML, see :mod:`energizer.backpack`.

``"gpu"`` is accepted as a deprecated alias of ``"mlx"``.
"""

import warnings

import numpy as np
from energizer._mlx import mx, MLX_AVAILABLE

_DEVICE_ALIASES = {"gpu": "mlx"}


def normalize_device(device):
    """Return the canonical device string. ``None`` is passed through."""
    if device in _DEVICE_ALIASES:
        warnings.warn(
            f"device '{device}' is deprecated, use '{_DEVICE_ALIASES[device]}': "
            "this backend is MLX/Metal, not the Neural Engine.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _DEVICE_ALIASES[device]
    return device


class Backend:
    """
    Centralised dispatch layer.
    All tensor ops go through here — no backend-specific
    code should live inside Function or Module classes.
    """

    VALID_DEVICES = ("cpu", "mlx")

    # ── Core helpers ────────────────────────────────────────────

    @staticmethod
    def is_available(device: str) -> bool:
        device = normalize_device(device)
        if device == "mlx":
            return MLX_AVAILABLE
        return True

    @staticmethod
    def validate(device: str) -> str:
        """Check *device* and return its canonical name."""
        device = normalize_device(device)
        if device not in Backend.VALID_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. "
                f"Expected one of {Backend.VALID_DEVICES}"
            )
        if device == "mlx" and not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not installed or not available on this machine. "
                "Install it with: pip install mlx"
            )
        return device

    @staticmethod
    def lib(device: str):
        """Return the raw backend module (np or mx) for a given device."""
        device = Backend.validate(device)
        return mx if device == "mlx" else np

    # ── Array creation ──────────────────────────────────────────

    @staticmethod
    def array(data, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            return mx.array(data)
        return np.array(data, dtype=np.float32)

    @staticmethod
    def zeros(shape, device: str):
        device = Backend.validate(device)
        return mx.zeros(shape) if device == "mlx" else np.zeros(shape, dtype=np.float32)

    @staticmethod
    def ones(shape, device: str):
        device = Backend.validate(device)
        return mx.ones(shape) if device == "mlx" else np.ones(shape, dtype=np.float32)

    @staticmethod
    def randn(*shape, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            return mx.random.normal(shape)
        return np.random.randn(*shape).astype(np.float32)

    # ── Linear algebra ──────────────────────────────────────────

    @staticmethod
    def matmul(a, b, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            a = mx.array(a) if not isinstance(a, mx.array) else a
            b = mx.array(b) if not isinstance(b, mx.array) else b
            return mx.matmul(a, b)
        return np.matmul(a, b)

    @staticmethod
    def transpose(a, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            a = mx.array(a) if not isinstance(a, mx.array) else a
            return mx.transpose(a)
        return np.transpose(a)

    # ── Element-wise ops ────────────────────────────────────────

    @staticmethod
    def exp(a, device: str):
        device = Backend.validate(device)
        return mx.exp(a) if device == "mlx" else np.exp(a)

    @staticmethod
    def log(a, device: str):
        device = Backend.validate(device)
        return mx.log(a) if device == "mlx" else np.log(a)

    @staticmethod
    def sqrt(a, device: str):
        device = Backend.validate(device)
        return mx.sqrt(a) if device == "mlx" else np.sqrt(a)

    @staticmethod
    def clip(a, min_val, max_val, device: str):
        device = Backend.validate(device)
        return (
            mx.clip(a, min_val, max_val)
            if device == "mlx"
            else np.clip(a, min_val, max_val)
        )

    # ── Reductions ──────────────────────────────────────────────

    @staticmethod
    def sum(a, axis=None, keepdims=False, device: str = "cpu"):
        device = Backend.validate(device)
        if device == "mlx":
            return (
                mx.sum(a, keepdims=keepdims)
                if axis is None
                else mx.sum(a, axis, keepdims=keepdims)
            )
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(a, axis=None, keepdims=False, device: str = "cpu"):
        device = Backend.validate(device)
        if device == "mlx":
            return (
                mx.mean(a, keepdims=keepdims)
                if axis is None
                else mx.mean(a, axis, keepdims=keepdims)
            )
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(a, axis=None, keepdims=False, device: str = "cpu"):
        device = Backend.validate(device)
        if device == "mlx":
            return (
                mx.max(a, keepdims=keepdims)
                if axis is None
                else mx.max(a, axis, keepdims=keepdims)
            )
        return np.max(a, axis=axis, keepdims=keepdims)

    # ── Activations ─────────────────────────────────────────────

    @staticmethod
    def relu(a, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            return mx.maximum(a, mx.zeros_like(a))
        return np.maximum(a, 0)

    @staticmethod
    def sigmoid(a, device: str):
        device = Backend.validate(device)
        if device == "mlx":
            return mx.sigmoid(a)
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    def softmax(a, axis=-1, device: str = "cpu"):
        device = Backend.validate(device)
        if device == "mlx":
            return mx.softmax(a, axis=axis)
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    # ── Type / device conversion ─────────────────────────────────

    @staticmethod
    def to_numpy(a, device: str) -> np.ndarray:
        """Convert any backend array to a NumPy array (e.g. for logging)."""
        device = normalize_device(device)
        if device == "mlx":
            return np.array(a)
        return np.asarray(a)

    @staticmethod
    def transfer(a, from_device: str, to_device: str):
        """Move raw data between backends."""
        from_device = normalize_device(from_device)
        to_device = normalize_device(to_device)
        if from_device == to_device:
            return a
        raw = Backend.to_numpy(a, from_device)
        return Backend.array(raw, to_device)


backend = Backend()
