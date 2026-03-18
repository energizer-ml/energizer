from __future__ import annotations

import os


def _mlx_disabled() -> bool:
    return os.environ.get("ENERGIZER_DISABLE_MLX", "").lower() in {"1", "true", "yes"}


if _mlx_disabled():
    mx = None
    MLX_AVAILABLE = False
else:
    try:
        import mlx.core as mx

        MLX_AVAILABLE = True
    except Exception:
        mx = None
        MLX_AVAILABLE = False
