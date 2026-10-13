# Energizer

> A small PyTorch-like deep-learning library for Apple Silicon, used as a testbed
> to study **what the Apple Neural Engine (ANE) can and cannot do for training**.

[![Lint](https://github.com/fgrimaepitech/energizer/actions/workflows/lint.yml/badge.svg)](https://github.com/fgrimaepitech/energizer/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains

1. **`energizer/`** – the library: autograd, layers, optimizers, an MLX/Metal
   backend, a Core ML compiler (`backpack`) and a data-parallel layer (`parallel`).
2. **`experiments/`** – the measurement harness behind the paper. Every number is
   regenerated from raw JSON Lines in `results/` by a script.
3. **`paper/`** – manuscript sources and generated tables.
4. **`docs/`** – research plan, hardware/software claims, API reference.

## What runs where (important)

| Path in this repo | Silicon actually used | Notes |
|---|---|---|
| `device="cpu"` | CPU | NumPy |
| `device="mlx"` | **GPU** (Metal) | MLX. *Not* the ANE. Formerly called `"gpu"`, kept as deprecated alias. |
| `energizer.backpack` (Core ML) | CPU / GPU / **ANE** | Inference only. Core ML decides placement; see `docs/claims.md`. |

MLX has no ANE backend. The ANE is only reachable through Core ML, which exposes
no public training or backward-pass API; Core ML also gives no control over
placement beyond `MLComputeUnits`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests
```

```python
import energizer as ez

model = ez.Sequential(ez.Linear(784, 256), ez.ReLU(), ez.Linear(256, 10)).to("mlx")
x = ez.Tensor.randn(32, 784, device="mlx")
loss = ez.CrossEntropyLoss()(model(x), ez.Tensor.zeros((32,), device="mlx"))
loss.backward()
```

Full API: [`docs/api.md`](docs/api.md).

## Reproducing the results

```bash
make smoke        # a few configs, ~minutes, validates the toolchain
make sweep        # placement / latency sweep (step 3)
sudo -E make power  # same sweep + powermetrics (needs root)
make baselines    # MLX vs PyTorch-MPS vs Core ML (step 4)
make tables       # results/*.jsonl -> paper/tables/*.tex
```

Each result file starts with an `env` record (chip, macOS build, library
versions, git commit, dirty flag) so a number can always be traced to the setup
that produced it. Requirements: Apple Silicon Mac, macOS 14.4+ (for
`MLComputePlan`), Python 3.10-3.12. Power needs `sudo` for `powermetrics`.

## Research plan

See [`docs/research_plan.md`](docs/research_plan.md) for the ordered work items
(backend claim fix, harness, sweeps, baselines, MLX-native backward, hybrid
ANE-forward / GPU-backward prototype, private-API decision) and their status.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).

## Author

Florian GRIMA – [@fgrimaepitech](https://github.com/fgrimaepitech)
