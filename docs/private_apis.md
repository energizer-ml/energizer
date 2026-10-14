# Private-API decision record

**Status: no private APIs used. Decision: do not use them yet.**

Rule: use private (undocumented) Apple interfaces only if the public path blocks a
measurement that the paper needs, and document the exact OS version when we do.

## Why the public path is enough so far

| Need | Public mechanism | Used in |
|---|---|---|
| Where does each op run | `MLComputePlan` (coremltools `compute_plan`) | `experiments/ezbench/coreml_utils.py` |
| Restrict placement | `MLComputeUnits` (`ALL`, `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`) | sweeps, baselines |
| Power per rail (CPU / GPU / ANE) | `powermetrics` (root) | `ezbench/power.py` |
| GPU training path | MLX / Metal | `energizer` `"mlx"` device |
| ANE forward, GPU backward | Core ML forward + MLX backward | `experiments/run_hybrid.py` |

## What the public path cannot do (candidates for private APIs)

- Run a backward pass or weight update *on the ANE*. Core ML ML Programs are
  immutable at runtime; new weights require re-conversion + compile + load
  (measured in `run_hybrid.py`, hundreds of ms for a small MLP).
- Force a specific op onto the ANE; Core ML plans placement itself.

## Revisit when

`run_hybrid.py` shows the forward gain is large enough that the weight-sync cost
is the only blocker, **and** a measured ANE-resident update would remove it.
Before that, pursue public mitigations first: stale-weight sync every K steps,
LoRA-style small adapters recompiled instead of the full model, fp16 only.

## If used, record here

- macOS version / build and chip the interface was verified on (reference machine:
  Apple M4 Pro, macOS 26.6.2, build 25G83)
- the exact symbol / framework used and how it was discovered
- fallback behaviour on other OS versions
- the claim in `docs/claims.md` it supports
