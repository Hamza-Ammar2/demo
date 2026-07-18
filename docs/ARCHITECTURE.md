# Architecture

## Design invariant
**The LLM never computes a finding.** Every `Finding` is produced by deterministic,
inspectable code and carries provenance. LLMs are confined to (a) extracting candidate
events from free text and (b) phrasing plain-language explanations — both validated against
the schema and the safety guard.

## Layers

1. **Schema** (`cyclebench/schema.py`) — Pydantic v2 entities + validators; JSON Schema
   export (`docs/schema/`). The contract every other layer speaks.
2. **Ingestion** — `Case` aggregates subject + events + cycles + sources in memory
   (`cyclebench/case.py`). Optional LLM extraction produces schema-valid events.
3. **Deterministic engine** (`cyclebench/engine/`):
   - `timeline.py` — chronological ordering, date-precision handling, undated-last.
   - `cycle.py` — cycle-length estimation, regularity, phase assignment with
     **mode-aware** logic (retrospective may use a later onset; causal may not).
   - `patterns.py` — cyclical clustering + change-after-event, returning raw metrics.
   - `confounders.py` — sleep/stress/medication overlap detection.
   - `missing.py` — critical data-gap detection.
   - `pipeline.py` — orchestrates all of the above into `Finding`s + a `DoctorBrief`, then
     runs `assert_brief_safe`.
4. **Grounding & evaluation**:
   - `audit.py` — CycleBench-Audit (10 leakage/safety assertions).
   - `benchmark/` — cases, baselines, runner → `results/`.
   - `adapters/` — mcPHASES aggregate validation, NHANES harmonizer, dataset registry.
5. **Delivery**:
   - `api/main.py` — FastAPI; in-memory, non-persistent; serves the frontend.
   - `web/index.html` — static "Doctor Mode" UI (no build step; robust for demo).

## Analysis modes (leakage boundary)
- **retrospective** — reconstruct the past using all data, including later period onsets.
- **causal** — only information available *as of* each event's `observed_at`; forbids
  `next_known_onset` and future data at the schema level. This is what CycleBench-Audit
  enforces, and what makes prediction claims trustworthy.

## Determinism & reproducibility
Pinned deps (`requirements.txt`), no randomness in the engine, snapshot/golden tests, and
`evaluate()` proven identical across runs. `make test` runs the scientific core first.
