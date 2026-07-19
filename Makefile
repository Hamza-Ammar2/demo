PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-pfl pfl-smoke pfl-full test demo benchmark audit mcphases nhanes reference foundation foundation-demo train-models train-tasks model-demo api clean help

help:
	@echo "Aestra / CycleBench"
	@echo ""
	@echo "  Offline core (no API keys, no restricted data):"
	@echo "    make install       - create venv + install deps + editable package"
	@echo "    make test          - run pytest"
	@echo "    make demo          - print Sarah's doctor brief"
	@echo "    make audit         - CycleBench-Audit (leakage fixture rejected)"
	@echo "    make benchmark     - CycleBench-Bench v0.1 → results/"
	@echo "    make api           - Aestra UI + API at http://127.0.0.1:8000"
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
	@echo ""
	@echo "  Optional pFL research (needs torch + local mcPHASES):"
	@echo "    make install-pfl   - pip install torch matplotlib"
	@echo "    make pfl-smoke     - multi_symptom, PFL_ROUNDS=3 → results/"
	@echo "    make pfl-full      - multi_symptom, 30 rounds (slow)"
	@echo ""
	@echo "  make clean           - remove caches"

install:
	test -d .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -r requirements.txt
	$(PIP) install -e . >/dev/null

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

install-pfl:
	$(PIP) install 'torch>=2.2' matplotlib

pfl-smoke:
	PFL_ROUNDS=3 $(PY) -m federated_learning.run multi_symptom

pfl-full:
	PFL_ROUNDS=30 $(PY) -m federated_learning.run multi_symptom

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
