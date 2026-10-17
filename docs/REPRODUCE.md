# Reproducing every table and figure

1. Recreate the environment (`environment/README.md`). Use the same macOS /
   Xcode / chip, or say so in the paper.
2. Plug the machine into AC power, disable Low Power Mode, close other apps,
   let it cool. `ezbench.env` records power source and thermal state in every
   result file; check `git_dirty` is `false` (commit first).
3. One command for everything that does not need root:

   ```bash
   make reproduce
   ```

4. Power / energy needs root for `powermetrics`; idle power is measured first and
   subtracted:

   ```bash
   sudo -E make power
   sudo -E python experiments/run_convergence.py --power   # energy-to-accuracy
   sudo -E make frozen-power                               # frozen-trunk energy
   ```

5. Build the paper and refresh the website copy: `make paper`.
6. Outputs: `results/<experiment>/*.jsonl` (raw), `paper/tables/*.tex`,
   `paper/figures/*.pdf` (derived, git-ignored, always regenerated).

## Experiment to paper-claim map

| Script | Claim / paper element |
|---|---|
| `run_placement_sweep.py` | ANE placement, latency, power vs channels / seq / batch / precision |
| `run_baselines.py` | MLX vs PyTorch MPS vs Core ML latency |
| `run_peak.py` | measured peak -> achieved-vs-peak efficiency |
| `run_train_step.py` | MLX-native training step vs NumPy |
| `run_ablation.py` | which MLX-native optimization matters; model-size sweep |
| `run_hybrid.py` | overhead breakdown: compute / transfer / recompilation / update |
| `run_convergence.py` | correctness (same accuracy) and time- / energy-to-accuracy |
| `run_frozen.py` | frozen ANE trunk + GPU-trained head: time, energy, accuracy |

## Variance and warm-up

`timing.measure` does untimed warm-up then >= `iters` timed runs; `timing.repeat`
repeats whole measurements to report run-to-run spread (`across_std_ms`). Report
median and spread, never a single run.

## Checklist before quoting a number

- `git_dirty` is `false` and `git_commit` is the commit you cite.
- `on_ac` is `true`, `low_power_mode` is `false`, `thermal_note` is empty.
- `python`, `numpy`, `mlx`, `torch`, `coremltools` in the env record match
  `environment/requirements.lock`.
- The sanity gate holds on a quiet machine: MLX fp16 4096 matmul, batch 1 is about 23 ms
  (a 3-4x slower value means the machine was busy; discard the run).
- Run each experiment at least twice and report the spread.
