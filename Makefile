PY ?= python
EXP = experiments

.PHONY: ablation test smoke sweep power baselines peak train hybrid convergence tables figures reproduce lint

test:
	$(PY) -m pytest -q tests

smoke:
	cd $(EXP) && $(PY) run_placement_sweep.py --preset smoke --iters 20
	cd $(EXP) && $(PY) run_baselines.py --preset smoke --iters 20

sweep:        # placement + latency (step 3)
	cd $(EXP) && $(PY) run_placement_sweep.py --preset full

power:        # run as: sudo -E make power  (placement + latency + power, idle-subtracted)
	cd $(EXP) && $(PY) run_placement_sweep.py --preset full --power

baselines:    # MLX vs PyTorch MPS vs Core ML (step 4)
	cd $(EXP) && $(PY) run_baselines.py --preset full

peak:         # measured peak throughput per backend
	cd $(EXP) && $(PY) run_peak.py

train:        # Energizer training-step latency, CPU vs MLX
	cd $(EXP) && $(PY) run_train_step.py

hybrid:       # ANE-forward / GPU-backward overhead breakdown
	cd $(EXP) && for w in 256 512 1024; do for u in ALL CPU_AND_NE; do \
	  $(PY) run_hybrid.py --width $$w --units $$u || exit 1; done; done

ablation:     # each MLX-native optimization off in turn + model-size sweep
	cd $(EXP) && $(PY) run_ablation.py

convergence:  # same loss/accuracy on cpu, mlx and hybrid-fp16 paths
	cd $(EXP) && $(PY) run_convergence.py

tables:
	cd $(EXP) && $(PY) make_tables.py

figures:
	cd $(EXP) && $(PY) make_figures.py

reproduce: test sweep baselines peak train ablation hybrid convergence tables figures

lint:
	black --check energizer tests experiments examples scripts
