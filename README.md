# Aestra — powered by CycleBench

> **Aestra** turns symptoms, cycles, and changes into an appointment-ready soft read.
> Underneath, **CycleBench** is the open scientific stack: schema, deterministic engine,
> leakage audit, benchmark, medical foundation graph, and specialist models.

Built for the Hack-Nation challenge
*[Open AI Infrastructure for Women's Hormonal Health](challenge.txt)* —
Tracks **01** (data & benchmark) + **02** (models) + **03** (application).

**We are not claiming to be “the world’s first” infrastructure alone.** We contribute
**reusable open building blocks** others can extend: schema, evidence, models, and an app
that uses them **without inventing medicine**.

---

## Judge checklist (reproducible offline)

No API keys and no restricted datasets are required for this path:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install
make test                          # ~78 tests
make demo                          # Sarah doctor brief (CLI)
make audit                         # leakage fixture rejected
make benchmark                     # writes results/benchmark_*.json
make api                           # http://127.0.0.1:8000  → Aestra UI
```

Optional (need local data / credentials — see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)):

| Command | Needs |
|---------|--------|
| `make train-models` | mcPHASES for hormonal-state; menopause falls back to **synthetic SWAN-like** |
| `make train-tasks` | Kaggle PCOS under `data/kaggle/` |
| `make mcphases` | Restricted PhysioNet CSVs (never redistributed) |
| `make foundation` | Rebuilds `data/foundation/foundation_v0.1.json` from seed + local evidence |
| `make install-pfl` / `make pfl-smoke` | PyTorch + `models/global_pfl_model.pt` (personalized phase) |

**fed-merge-finished:** Aestra UI from `finished` + Model-pFL → LLM doctor follow-ups.
See [docs/FED_MERGE.md](docs/FED_MERGE.md) and [docs/PFL.md](docs/PFL.md).

**For judges / videos:** [docs/JUDGE_CARD.md](docs/JUDGE_CARD.md) (honest numbers + track map) ·
[`logic.md`](logic.md) (ChatGPT / narration brief).

**Host the demo:** [docs/DEPLOY.md](docs/DEPLOY.md) (Render / Railway / Docker).

---

## 20-second pitch

Generic chatbots can *summarize* a health history, but they cannot reliably reconstruct a
timeline, separate observation from inference, test whether a pattern repeats, flag
confounders, prevent future-information leakage, or link every claim to evidence.
**CycleBench does that deterministically.** The optional LLM only translates language; it
never computes a finding. **Aestra** is the application layer: chip-based intake → soft read
for a doctor visit. A built-in **leakage audit** catches the trick that makes naive models
look smarter than they are: reading the future.

---

## What's reusable (the open asset)

| Asset | Where | Reuse |
|-------|-------|-------|
| **CycleBench schema** | `cyclebench/schema.py`, `docs/schema/*.json` | model longitudinal hormonal-health cases |
| **Deterministic engine** | `cyclebench/engine/` | timeline, cycle alignment, patterns, confounders, missing info |
| **CycleBench-Audit** | `cyclebench/audit.py` | 10 leakage/safety assertions |
| **CycleBench-Bench v0.1** | `cyclebench/benchmark/`, `results/` | documented cases + metrics |
| **Medical foundation graph** | `cyclebench/foundation/`, `data/foundation/` | seeded associations + dataset evidence |
| **Harmonized NHANES** | `data/nhanes_harmonized/` | open (CC-BY-4.0) female hormone ranges |
| **Specialist models** | `models/`, `results/model_*.json` | hormonal-state, menopause-stage*, PCOS-risk |
| **Aestra web app** | `web/`, `api/` | Track 03 demo of the stack |

\*Menopause-stage checkpoint may be trained on **synthetic SWAN-like** data until real ICPSR
SWAN is loaded — see `docs/MODEL_CARD.md` and `docs/SWAN_ACCESS.md`. Always check
`results/model_menopause_stage.json` → `data_source`.

---

## Architecture

```
 intake / free text ─▶ [optional LLM extract]
                            │
                            ▼
              CycleBench Case (schema-valid)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  deterministic        foundation           specialist
  engine               assemble_read        models
  (patterns)           (fact+evidence)      (signals)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   DoctorBrief / soft read
                   + CycleBench-Audit
```

---

## Models & data honesty

| Piece | Training / source | Ships in repo? |
|-------|-------------------|----------------|
| Hormonal-state | mcPHASES (restricted) | Checkpoint may ship; **raw rows never** |
| PCOS-risk | Kaggle Kerala cohort | Checkpoint may ship; raw download gitignored |
| Menopause-stage | SWAN if present, else **synthetic_swan_like** | Checkpoint + metrics with `data_source` label |
| NHANES harmonized | CDC public | **Yes** — `data/nhanes_harmonized/` |
| mcPHASES raw | PhysioNet DUA | **No** — gitignored |
| Foundation bundle | Seed + adapters | **Yes** — `data/foundation/` |

---

## Limitations & medical safety

This is **not** a diagnostic tool and **not** a medical device. It surfaces associations for
conversation with a clinician. See `docs/LIMITATIONS.md` and `docs/MEDICAL_SAFETY.md`.

We do **not** claim clinical menopause-onset prediction. The menopause **stage category**
model is research/demo infrastructure and must disclose synthetic training when applicable.

---

## Repository layout

```
cyclebench/     schema, engine, audit, benchmark, foundation, models, adapters, cli
api/            FastAPI (serves web/ + JSON endpoints)
web/            static Aestra UI (no Node build)
data/           nhanes_harmonized/ + foundation/ committed; raw/restricted gitignored
docs/           architecture, reproducibility, model card, datasets, …
results/        benchmark + model metrics (aggregate)
tests/          pytest suite
models/         trained checkpoints (joblib) when present
```

---

## Open science & license

- Code: **MIT**
- Docs + harmonized NHANES export: **CC-BY-4.0**
- mcPHASES raw data: **not included** (restricted)

See `LICENSE`, `CITATION.cff`, [CONTRIBUTING.md](CONTRIBUTING.md), and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
