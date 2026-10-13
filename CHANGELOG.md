# Changelog

## 0.2.0 — research release

### Breaking / renamed
- Device `"gpu"` is now `"mlx"` (Apple GPU via Metal; never the Neural Engine).
  `"gpu"` and `Tensor.gpu()` still work and emit a `DeprecationWarning`.
- `pip install "energizer[mlx]"` replaces `[gpu]` (alias kept).
- `Conv1d`/`Conv2d` weights/bias now take part in autograd; `LayerNorm` /
  `BatchNorm*` scale and shift are registered `Parameter`s (they were not, so
  optimizers never updated them) and `BatchNorm1d` gamma starts at 1.

### Added
- MLX-native backward for every layer: Conv1d/2d, ConvTranspose2d, Max/AvgPool2d,
  LayerNorm, BatchNorm1d/2d, Dropout, Embedding, MultiheadAttention /
  TransformerEncoder; MLX-native cross-entropy, SGD and Adam. Gradients are
  checked against PyTorch on CPU and MLX.
- `Tensor.mean(axis, keepdims)`, `permute`, `transpose(d0, d1)`, reflected
  operators (`2 * t`, `1 / t`).
- `experiments/` measurement harness (placement via `MLComputePlan`, latency,
  `powermetrics` power with idle subtraction, baselines, peak, ablations,
  hybrid ANE-forward / GPU-backward prototype, convergence), `make reproduce`.
- `docs/` research plan, claims ledger, private-API decision record,
  reproducibility guide; `paper/` LaTeX skeleton; `CITATION.cff`.

### Fixed
- `SGD(model.parameters())` was a silent no-op on CPU and an infinite loop on MLX.
- Batched `MatMul` backward transposed all axes instead of the last two.
- `Tensor.mean(axis=...)` ignored its axis.

### Repo
- Tracked `egg-info`, stale `requirements.txt` removed; legacy scripts moved to
  `examples/legacy/`, `update_version.py` to `scripts/`.
