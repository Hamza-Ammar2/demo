# Case Compiler — powered by CycleBench

> Sarah has ten minutes to explain six months of changing symptoms. **Case Compiler**
> turns fragmented symptoms, menstrual history, medication changes, sleep, and optional
> health measurements into a traceable doctor brief. Underneath, **CycleBench** provides an
> open schema, benchmark, and leakage-audited analysis pipeline so every finding can be
> inspected and reproduced.

Built for the Hack-Nation *Open AI Infrastructure for Women's Hormonal Health* challenge.
Our contribution is **Track 01 (data & benchmark infrastructure) + Track 03 (application)**:
a reusable scientific asset, not just a demo.

---

## 20-second pitch

Generic chatbots can *summarize* a health history, but they cannot reliably reconstruct a
timeline, separate observation from inference, test whether a pattern repeats, flag
confounders, prevent future-information leakage, or link every claim to evidence.
Case Compiler does — **deterministically**. The LLM only translates language; it never
computes a finding. And a built-in **leakage audit** catches the exact trick that makes
naive models look smarter than they are: reading the future.

## What's reusable (the open asset)

| Asset | Where | Reuse |
|-------|-------|-------|
| **CycleBench schema** (6 entities) | `cyclebench/schema.py`, `docs/schema/*.json` | model longitudinal hormonal-health cases in any language |
| **Deterministic engine** | `cyclebench/engine/` | timeline, cycle alignment (retrospective/causal), pattern + confounder + missing-info detection |
| **CycleBench-Audit** | `cyclebench/audit.py` | 10 leakage/safety assertions for any cycle prediction task |
| **CycleBench-Bench v0.1** | `cyclebench/benchmark/`, `results/` | 10 documented cases + metrics for doctor-brief quality |
| **Harmonized NHANES dataset** | `data/nhanes_harmonized/` | open (CC-BY-4.0) female hormone reference ranges |
| **mcPHASES validation** | `results/mcphases_validation.json` | aggregate real-data grounding (no raw data redistributed) |

## Architecture

```
 raw history ─▶ [LLM extract*]         *optional; demo runs offline
                     │  (candidate events, validated against schema)
                     ▼
   ┌──────────────────────────────────────────────────────────┐
   │  CycleBench deterministic engine                           │
   │  timeline ─▶ cycle alignment (retrospective|causal)        │
   │           ─▶ pattern detector ─▶ confounders ─▶ missing    │
   │           ─▶ provenance ─▶ Findings ─▶ DoctorBrief         │
   └──────────────────────────────────────────────────────────┘
        │                    │                      │
   CycleBench-Audit    CycleBench-Bench       FastAPI + web
   (leakage/safety)    (evaluation)           (Doctor Mode UI)
```

## Quick start

```bash
make install     # venv + pinned deps + editable package
make test        # full test suite (scientific core first)
make demo        # print Sarah's doctor brief (offline, no API key)
make audit       # prove the leaking split is rejected
make benchmark   # run CycleBench-Bench, write results/
make api         # FastAPI + web UI at http://localhost:8000
```

## Golden demo (Sarah)

`make demo` (or "Load Sarah demo" in the UI) compiles five cycles of migraines. The engine
discovers **4 of 5 severe migraines fall in the same (luteal) cycle window** in
*retrospective* mode — and honestly reports only **3 of 5** in *causal* mode, because the
earliest episode cannot be phase-assigned without peeking at a later period onset. Reduced
sleep is flagged as a confounder; the missing contraceptive formulation is surfaced; every
finding carries provenance; nothing is phrased as diagnosis or causation.

## Benchmark methodology

`CycleBench-Bench v0.1` compares a naive summarizer (Path A) against the CycleBench engine
(Path B) on 10 documented synthetic cases (positive / negative / misleading / irregular /
insufficient). See `docs/BENCHMARK.md` and `results/benchmark_results.json`. Headline
(reproducible via `make benchmark`): the naive path has an **83% false-pattern rate** and
makes unsupported causal claims; CycleBench has a **0% false-pattern rate, 100% provenance
coverage, and 0 safety violations**.

## Datasets

- **mcPHASES** (PhysioNet, **restricted**) — used locally for aggregate validation only;
  never redistributed. See `docs/DATASETS.md` for access instructions.
- **NHANES** (CDC, **public domain**) — a harmonized female-hormone export is published in
  `data/nhanes_harmonized/` (CC-BY-4.0).
- `docs/DATASET_REGISTRY.md` documents what each source can and cannot validly support.

## Limitations & medical safety

This is **not** a diagnostic tool. It detects associations, never causation. See
`docs/LIMITATIONS.md` and `docs/MEDICAL_SAFETY.md`. We do **not** predict hormone levels or
menopause (the data does not support it — see LIMITATIONS); we detect and audit *patterns*.

## Repository contents

```
cyclebench/    schema, engine, audit, benchmark, adapters, fixtures, cli
api/           FastAPI backend
web/           static Doctor Mode frontend
data/          gitignored raw data; data/nhanes_harmonized/ is the open export
docs/          DATA_MODEL, BENCHMARK, DATASETS, LIMITATIONS, MEDICAL_SAFETY,
               ARCHITECTURE, DATASHEET, DEMO, DECISIONS, schema/*.json
results/       benchmark + mcPHASES aggregate metrics
tests/         53 tests (scientific core, audit, benchmark, API)
```

## Open-science contribution & license

Code: **MIT**. Docs and the harmonized NHANES export: **CC-BY-4.0**. mcPHASES raw data is
**not** included (restricted). See `LICENSE` and `CITATION.cff`.
