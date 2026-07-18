PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install test demo benchmark audit mcphases nhanes train-models model-demo api dev clean help

help:
	@echo "CycleBench / Case Compiler"
	@echo "  make install       - create venv + install pinned deps + editable package"
	@echo "  make test          - run pytest (scientific core first)"
	@echo "  make demo          - print Sarah's doctor brief (offline, no API key)"
	@echo "  make train-models  - train Layer 02 models (hormonal-state + menopause-stage)"
	@echo "  make model-demo    - demo multi-source + menopause predictions with explanations"
	@echo "  make benchmark     - run CycleBench-Bench v0.1, save real metrics to /results"
	@echo "  make audit         - run CycleBench-Audit; prove the leaking fixture is rejected"
	@echo "  make mcphases      - real mcPHASES validation (aggregate output only)"
	@echo "  make nhanes        - build harmonized open NHANES dataset export"
	@echo "  make api           - run FastAPI backend on :8000"
	@echo "  make dev           - run web frontend (Tier C)"

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

api:
	.venv/bin/uvicorn api.main:app --reload --port 8000

dev:
	cd web && npm run dev

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
