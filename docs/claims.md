# Claims ledger

What the paper may and may not claim, and the evidence for each. Update this file
whenever a claim changes; every claim must point to a script and a result file.

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | The `"mlx"` backend executes on the Apple GPU (Metal), not on the ANE. | Stated by MLX docs; code/docs corrected | `energizer/backend.py` |
| C2 | Core ML may place ops on the ANE; placement depends on op type, shapes and precision. | To measure | `experiments/run_placement_sweep.py` |
| C3 | ANE power/latency vs GPU for the same op. | To measure | `--power` sweep (needs `sudo`) |
| C4 | Core ML exposes no public backward pass, so ANE training needs a workaround. | Literature/docs check | to cite |
| C5 | MLX-native backward on Energizer trains faster than the NumPy/mixed path. | Implemented, gradient-checked vs PyTorch; timing on `run_train_step.py` (clean-commit run pending) | `experiments/run_train_step.py`, `tests/test_layers_autograd.py` |
| C6 | Hybrid (ANE forward, GPU backward) pays off. | Not prototyped | roadmap item 6 |

## Threats to validity (keep honest)

- `MLComputePlan` reports the *planned* device. The OS may still schedule
  differently at runtime; validate plan vs. observed ANE power (`powermetrics`).
- Single machine / OS build per run. Always report the `env` record of the result file.
- Thermal state and low-power mode change latency; record and control them.
- Private APIs (if used) are tied to one macOS version; document it.
