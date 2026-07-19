PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-pfl test demo benchmark audit mcphases nhanes reference foundation foundation-demo \
	train-models train-tasks model-demo api clean help pfl-smoke pfl-train-global pfl-train-local pfl-sync

help:
	@echo "Aestra / CycleBench (fed-merge-finished: Aestra UI + personalized phase pFL)"
	@echo ""
	@echo "  Offline core (no API keys, no restricted data):"
	@echo "    make install       - create venv + install deps + editable package"
	@echo "    make install-pfl   - PyTorch extras for personalized phase model"
	@echo "    make test          - run pytest"
	@echo "    make demo          - print Sarah's doctor brief"
	@echo "    make audit         - CycleBench-Audit (leakage fixture rejected)"
	@echo "    make benchmark     - CycleBench-Bench v0.1 → results/"
	@echo "    make api           - Aestra UI + API at http://127.0.0.1:8000"
	@echo "    make pfl-smoke     - one pad×5 phase inference"
	@echo ""
	@echo "  Scientific rebuild (some need local data — see docs/REPRODUCIBILITY.md):"
	@echo "    make train-models  - hormonal-state + menopause-stage"
	@echo "    make train-tasks   - model-factory tasks (e.g. PCOS-risk)"
	@echo "    make model-demo    - explainable model predictions"
	@echo "    make foundation    - rebuild medical foundation graph"
	@echo "    make foundation-demo"
	@echo "    make reference     - aggregate corpus reference stats"
	@echo "    make nhanes        - rebuild open NHANES harmonized export"
	@echo "    make mcphases      - restricted PhysioNet aggregate validation"
	@echo ""
	@echo "  make clean           - remove caches"
	@echo "  See docs/PFL.md and docs/FED_MERGE.md"

install:
	test -d .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -r requirements.txt
	$(PIP) install -e . >/dev/null

install-pfl: install
	$(PIP) install -r requirements-pfl.txt

test:
	$(PY) -m pytest

demo:
	$(PY) -m cyclebench.cli demo

train-models:
	$(PY) -m cyclebench.model.train

train-tasks:
	$(PY) -m cyclebench.model.tasks

model-demo:
	$(PY) -m cyclebench.cli model-demo

benchmark:
	$(PY) -m cyclebench.benchmark.runner

audit:
	$(PY) -m cyclebench.cli audit

mcphases:
	$(PY) -m cyclebench.adapters.mcphases_validate

nhanes:
	$(PY) -m cyclebench.adapters.nhanes_harmonize

reference:
	$(PY) -m cyclebench.reference

foundation:
	$(PY) -m cyclebench.foundation build

foundation-demo:
	$(PY) -m cyclebench.foundation demo

api:
	.venv/bin/uvicorn api.main:app --reload --port 8000

pfl-smoke:
	$(PY) -c "from cyclebench.model.pfl import run_pfl_inference; \
import json; print(json.dumps(run_pfl_inference({\
'headaches_ord':2,'cramps_ord':4,'sorebreasts_ord':1,'fatigue_ord':3,'sleepissue_ord':2,\
'moodswing_ord':2,'stress_ord':3,'foodcravings_ord':2,'indigestion_ord':1,'bloating_ord':2,\
'appetite_ord':2,'sleep_minutes':380,'sleep_awake':20,'sleep_efficiency':0.9,'resting_hr':68,\
'steps_sum':7000,'stress_score_mean':40,'wrist_temp_delta':0.1,'glucose_mean':90,'hrv_rmssd':45,\
'is_weekend':0}), indent=2))"

pfl-train-global:
	$(PY) -c "from cyclebench.model.pfl import train_global_pfl_model; import json; print(json.dumps(train_global_pfl_model(epochs=10), indent=2))"

pfl-train-local:
	$(PY) -c "from cyclebench.model.pfl import train_local_pfl; import json; print(json.dumps(train_local_pfl(), indent=2))"

pfl-sync:
	$(PY) -c "from cyclebench.model.pfl import federated_sync_pfl; import json; print(json.dumps(federated_sync_pfl(peer_epochs=1, max_peers=5), indent=2))"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
