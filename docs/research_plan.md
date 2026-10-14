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

### Findings (M4 Pro, macOS 26.6.2, clean commits (`ead845c`, `c5e222a`, `d2c86f2`), medians; battery/contention caveats below)

- **Placement (576 rows, 0 errors):** fp32 never reaches the ANE (0/72 configs). fp16: 50/72 fully on the ANE, 3 partial, 19 none; the switch is a size threshold (conv ~128 channels at 64x64, transformer block seq 128 at d=512). `CPU_AND_NE` places 57/72 fp16 configs fully.
- **Baselines (fp16 matmul, batch 1):** Core ML wins at n<=128 (0.04-0.05 ms), PyTorch MPS at 512 (0.17 ms), MLX at 1024 and 4096 (0.51 / 22.9 ms).
- **Measured peak, fp16, n<=4096:** MLX 6.0, MPS 5.5, Core ML ANE 5.2, Core ML GPU 4.0 TFLOP/s (3 repeats within 3%). An earlier run (00:41 UTC) was 3-4x slow on GPU paths, with something else using the GPU; archived in `results/_archive/contaminated/`.
- **Training step vs NumPy (Adam):** MLP-1024 8.9 -> 1.9 ms, MLP-2048 40.1 -> 7.5 ms, CNN 35.6 -> 4.6 ms, transformer 29.7 -> 7.2 ms (4-8x).
- **Ablation (width 2048):** full 7.4 ms; host loss 7.7 ms; host optimizer 23.7 ms; NumPy 39.5 ms.
- **Hybrid overhead:** Core ML forward 0.21-0.47 ms, transfer 0.01-0.03 ms, GPU backward+SGD 0.57-0.89 ms vs pure MLX 0.45-1.08 ms; weight sync 260-350 ms. No width <=1024 where the hybrid step beats pure MLX even with free sync.
- **Convergence (MNIST MLP, 3 epochs):** NumPy 93.00%, MLX 93.02%, hybrid fp16 with weight refresh every step 92.98% but 135 s vs 0.2 s; refresh every 5 or 20 steps diverges (weights blow up to NaN).
- **Power (576 rows, `powermetrics`, idle 1.28 W CPU / 0.24 W GPU / 0 W ANE subtracted):** planned-ANE fp16 rows draw median 2.72 W on the ANE rail (108 rows); rows not planned on the ANE draw 0 W (460 rows), so planned placement matches observed ANE activity. For 31 batch-8 fp16 configs fully placed on the ANE, CPU+GPU took median 1.34x the latency and 2.82x the energy per inference. `ALL` sometimes splits a block (68-77% ANE) and is then slower than either pure choice.
- **Energy-to-accuracy (MNIST MLP, 3 epochs, ~93%):** NumPy 2.3 J / 0.4 s, MLX 2.1 J / 0.24 s, hybrid fp16 (refresh every step) 737 J / 135 s (CPU rail 690 J, ANE 63 J, GPU 0). The hybrid spends its energy recompiling on the CPU.
- **Caveats:** the power sweep and one convergence run report `Battery Power` in their env record although the machine was reportedly on AC. Large batch-8 matmuls (n>=1024) vary 1.4-5x between the two full sweeps (also fp32 CPU_ONLY), so quote medians over many configs, not single large-matmul rows.
- **Pending:** larger models, repeats across thermal states.

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
