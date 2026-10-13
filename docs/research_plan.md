# Research plan

Ordered work items. Status is updated as work lands; see `docs/claims.md` for the
claims each step supports.

| # | Step | Status | Where |
|---|---|---|---|
| 1 | Fix the backend claim: `"gpu"` -> `"mlx"`, docs say MLX/Metal, not ANE | **done** (alias `"gpu"` kept, emits `DeprecationWarning`) | `energizer/backend.py`, README, `docs/api.md` |
| 2 | Measurement harness: convert single conv / matmul / transformer block, log per-op placement (`MLComputePlan`) and power (`powermetrics`), scripted | **done**, smoke-validated on M4 Pro / macOS 26.6.2; power needs `sudo` run | `experiments/ezbench/`, `run_placement_sweep.py` |
| 3 | Characterisation sweeps: channels, sequence length, batch, precision -> placement, latency, power | harness ready, **full sweep not yet run** | `make sweep`, `sudo -E make power` |
| 4 | Baselines: MLX, PyTorch MPS, Core ML | script validated on smoke preset, **full run pending** | `run_baselines.py` |
| 5 | MLX-native backward pass in Energizer | **done**: every layer (Linear, activations, Conv1d/2d, ConvTranspose2d, Max/AvgPool2d, LayerNorm, BatchNorm1d/2d, Dropout, Embedding, MultiheadAttention / TransformerEncoder), cross-entropy, SGD and Adam run forward+backward on MLX with zero host transfers (checked by an audit that counts `np.array(mx.array)` calls); gradients verified against PyTorch on CPU and MLX | `energizer/layers/`, `energizer/autograd.py`, `tests/test_layers_autograd.py`, `tests/test_conv_autograd.py`, `experiments/run_train_step.py` |
| 6 | Hybrid prototype: ANE forward, GPU backward + optimizer; measure transfer / recompile overhead | **prototype done** (MLP only); sweep over widths / batch pending | `experiments/run_hybrid.py` |
| 7 | Private-API decision | **decided: not needed yet**, see record | `docs/private_apis.md` |

### Preliminary observations (single machine, dirty tree, n=30; NOT paper numbers)

Re-run on a clean commit before quoting. M4 Pro, macOS 26.6.2.

- Training step, MLP 3x1024, batch 256, Adam: MLX 6.2 ms before step 5 -> 2.0 ms after (3.2x); NumPy CPU 9.1 ms.
- Hybrid, same MLP: Core ML forward on ANE 0.47 ms (placement 100% ANE with `CPU_AND_NE`); GPU backward+SGD 0.90 ms; pure-MLX step 1.20 ms; weight sync (convert+compile+load) about 540 ms. Hybrid does not beat pure MLX at any sync interval tried.
- `compute_units=ALL` placed the same fp16 MLP on the GPU, not the ANE: placement is the planner's choice, which is why `CPU_AND_NE` is also swept.
- Transformer block (seq 64, d=128) fp16: 100% ANE under `ALL`; fp32: 0% ANE.

### Bugs found and fixed while doing step 5

- `SGD(model.parameters())` built one param group per tensor and iterated tensor rows: a silent no-op on CPU, an infinite loop on MLX. Fixed in `Optimizer`, regression test added.
- Missing reflected operators (`2 * t`, `1 / t`, ...). Added.
- Before step 5, `Conv1d`, `Conv2d`, `ConvTranspose2d`, pooling, `LayerNorm`, `BatchNorm*`, `Dropout` and `Embedding` produced **no gradient** (their forward built a fresh `Tensor`, cutting the graph); `BatchNorm1d`, pooling and `TransformerEncoderLayer` crashed on MLX; `LayerNorm`/`BatchNorm` scale/shift were not registered as parameters, so optimizers never updated them; `BatchNorm1d` initialised gamma to 0. All fixed.
- `MatMul` backward transposed *all* axes, wrong for batched (attention) matmuls. Now swaps the last two.
- `Tensor.mean(axis=...)` ignored `axis`; `Tensor.transpose` did not exist. Added `mean(axis, keepdims)`, `permute`, `transpose(d0, d1)`.

## Reproducibility rules

1. No hand-copied numbers: tables come from `results/` via `experiments/make_tables.py`.
2. Each results file embeds environment (chip, macOS build, library versions, git SHA, dirty flag).
3. Power rows are `null` unless `powermetrics` really ran; failures are recorded as `error` rows, never dropped.
4. Warm-up and iteration counts are in the script defaults and recorded per row.

## Suggested paper outline (`paper/main.tex`)

1. Introduction / what the ANE is and is not reachable from
2. Background: Core ML, MLX, MPS, ANE constraints
3. Methodology: harness, metrics, threats to validity
4. Characterisation results (step 3)
5. Baseline comparison (step 4)
6. Training path: MLX-native backward (step 5) and hybrid (step 6)
7. Discussion, private APIs, limitations
