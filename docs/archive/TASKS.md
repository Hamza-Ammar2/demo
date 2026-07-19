# TASKS — Case Compiler / CycleBench

## Repo state at build start
- `data/mcphases/` — full real mcPHASES (23 CSVs, 3.4 GB, verified). Gitignored (restricted).
- `data/raw/` + `data/processed/` — NHANES XPT + CSV (public domain). Gitignored (size); a
  harmonized slice will be published to `data/nhanes_harmonized/`.
- `SPEC.md` — build contract (this drives everything). `challenge.txt` — the hackathon brief.
- `.venv` — Python 3.14 with pydantic/pandas/numpy/scipy/sklearn/fastapi/pytest.
- Node v14 (too old for modern Next.js) — web tier will use a lightweight approach.

## Gap analysis (what exists vs. what SPEC needs)
| Area | Status | Plan |
|------|--------|------|
| Schema | none | Tier A — Pydantic + JSON Schema export |
| Deterministic engine | none | Tier A |
| Sarah offline demo | none | Tier A |
| Core tests | none | Tier A |
| Leakage audit | none | Tier B |
| Real mcPHASES validation | none | Tier B (data present) |
| Benchmark v0.1 | none | Tier B |
| NHANES harmonized dataset | raw only | Tier B |
| API | none | Tier C |
| Web | none | Tier C (node constraint) |
| Docs | partial (SPEC/challenge/thoughts) | throughout |

## Checklist
### Tier A — Minimum demoable slice
- [ ] `cyclebench/schema.py` (6 entities) + JSON Schema export
- [ ] `cyclebench/engine/` timeline, cycle, patterns, confounders, missing, provenance
- [ ] `cyclebench/fixtures/sarah.py` (pre-extracted, no LLM)
- [ ] `cyclebench/cli.py` → `make demo` prints doctor brief
- [ ] `tests/` core scientific tests green

### Tier B — Grounding + benchmark
- [x] `cyclebench/audit.py` (10 assertions) + leaking fixture rejected
- [x] `cyclebench/adapters/mcphases_validate.py` (aggregate real-data validation)
- [x] `cyclebench/benchmark/` cases + runner + `/results` real metrics
- [x] baselines: naive summarizer (Path A) vs deterministic engine (Path B)
- [x] `cyclebench/adapters/nhanes_harmonize.py` → `data/nhanes_harmonized/` + registry

### Tier C — API + web
- [x] `api/main.py` FastAPI endpoints (health, compile, demo/sarah, timeline, benchmark, audit)
- [x] `web/` golden path (landing + results + timeline), served by FastAPI

### Docs
- [x] README + DATA_MODEL, BENCHMARK, DATASETS, LIMITATIONS, MEDICAL_SAFETY,
      ARCHITECTURE, DATASHEET, DEMO, DECISIONS, DATASET_REGISTRY, CITATION.cff, schema/*.json

### Tier D — stretch (only after A+B green)
- [ ] LLM live extraction, model checkpoint, PDF export, polish

## Status: Tiers A–C + docs complete. 53 tests passing.

## Assumptions log → see docs/DECISIONS.md
