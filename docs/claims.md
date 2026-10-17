# Claims ledger

What the paper may and may not claim, and the evidence for each. Update this file
whenever a claim changes; every claim must point to a script and a result file.

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | The `"mlx"` backend executes on the Apple GPU (Metal), not on the ANE. | Stated by MLX docs; code/docs corrected | `energizer/backend.py` |
| C2 | Core ML may place ops on the ANE; placement depends on op type, shapes and precision. | Measured: 0/288 fp32 on ANE; fp16 ALL 50 full/3 partial/19 none | `experiments/run_placement_sweep.py` |
| C3 | ANE power/latency vs GPU for the same op. | Measured: ANE 2.7 W median when planned; GPU path 1.34x latency, 2.82x energy (31 configs) | `--power` sweep (needs `sudo`) |
| C4 | Core ML exposes no public backward pass, so ANE training needs a workaround. | Literature/docs check | to cite |
| C5 | MLX-native backward on Energizer trains faster than the NumPy/mixed path. | Implemented, gradient-checked vs PyTorch; timing 4-8x on clean commit (`run_train_step.py`) | `experiments/run_train_step.py`, `tests/test_layers_autograd.py` |
| C6 | Hybrid (ANE forward, GPU backward) pays off. | Refuted at widths 256-1024: never beats pure MLX; weight refresh 260-320 ms; K>1 diverges | `run_hybrid.py`, `run_convergence.py` |
| C7 | Frozen ANE trunk + GPU head beats an all-GPU path. | Confirmed: CNN trunks 2.1-2.4x faster, 10-11x less energy than MLX fp16, same accuracy; MLP 1.1-1.9x; pipelining +1.2-1.6x | `run_frozen.py`, `results/frozen/` |

## Threats to validity (keep honest)

- `MLComputePlan` reports the *planned* device. The OS may still schedule
  differently at runtime; validate plan vs. observed ANE power (`powermetrics`).
- Single machine / OS build per run. Always report the `env` record of the result file.
- Thermal state and low-power mode change latency; record and control them.
- Private APIs (if used) are tied to one macOS version; document it.
