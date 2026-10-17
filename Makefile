PY ?= python
EXP = experiments
export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: paper ablation test smoke sweep power baselines peak train hybrid convergence frozen tables figures reproduce lint

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

frozen:       # frozen ANE trunk + GPU-trained head (sudo -E make frozen-power for energy)
	cd $(EXP) && $(PY) run_frozen.py --preset full

frozen-power: # run as: sudo -E make frozen-power
	cd $(EXP) && $(PY) run_frozen.py --preset full --power

tables:
	cd $(EXP) && $(PY) make_tables.py

figures:
	cd $(EXP) && $(PY) make_figures.py

paper:        # build the PDF and refresh the copy served on the website
	cd paper && tectonic main.tex && cp main.pdf ../assets/energizer-ane-paper.pdf

reproduce: test sweep baselines peak train ablation hybrid convergence frozen tables figures

lint:
	black --check energizer tests experiments examples scripts
