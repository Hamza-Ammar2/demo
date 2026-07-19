# Reproducibility guide

This document is the single source of truth for **what works offline**, **what needs
extra data**, and **how to rebuild artifacts**.

## 0. Requirements

- Python **3.11+** (developed on 3.11–3.14)
- `make` (or run the equivalent `.venv/bin/python -m …` commands)
- No Node.js — the UI is static HTML/CSS/JS served by FastAPI

```bash
git clone <this-repo>
cd Hormones   # or your clone directory
python3 -m venv .venv
source .venv/bin/activate
make install
```

Copy `.env.example` → `.env` only if you need optional LLM / PhysioNet / Kaggle downloads.
**Never commit `.env`.**

## 1. Offline core (always)

These use fixtures + committed open artifacts only:

| Step | Command | Output |
|------|---------|--------|
| Tests | `make test` | pytest (~78) |
| Sarah demo | `make demo` | doctor brief on stdout |
| Leakage audit | `make audit` | PASS/FAIL |
| Benchmark | `make benchmark` | `results/benchmark_*.json` |
| Web UI | `make api` | http://127.0.0.1:8000 |

`make api` serves the Aestra UI from `web/` and the CycleBench JSON API from `api/`.

There is **no** `npm run dev`. Use `make api` only.

## 2. Committed scientific artifacts

Safe to redistribute (and expected in a clean clone):

- `data/nhanes_harmonized/` — open NHANES female hormone export
- `data/foundation/foundation_v0.1.json` — medical foundation bundle
- `results/*.json` — aggregate metrics (benchmark, model cards, validation)
- `models/*.joblib` — checkpoints when present (see honesty labels in `results/model_*.json`)
- `docs/` — methodology + model card + datasheets

## 3. Gitignored / never redistribute

| Path | Why |
|------|-----|
| `data/mcphases/` | PhysioNet restricted DUA |
| `data/kaggle/` | Third-party terms; re-download with your token |
| `data/swan/` | ICPSR user download |
| `data/raw/`, `data/processed/` | Large NHANES working copies |
| `data/qa/` | Local Q&A corpora |
| `data/contributions/` | User sessions from the app |
| `.env`, `.venv/` | Secrets + local env |

## 4. Optional rebuild commands

### Foundation graph

```bash
make foundation        # rebuild data/foundation/foundation_v0.1.json
make foundation-demo   # print a sample assembled read
```

### Models

```bash
make train-models   # hormonal-state (needs mcPHASES) + menopause-stage (SWAN or synthetic)
make train-tasks    # PCOS-risk (needs Kaggle PCOS CSV)
make model-demo     # CLI predictions + feature attributions
```

Check `data_source` in:

- `results/model_hormonal_state.json`
- `results/model_menopause_stage.json`  ← look for `synthetic_swan_like`
- `results/model_pcos_risk.json`

### Datasets

```bash
make nhanes      # rebuild harmonized export from local NHANES XPT/CSV
make mcphases    # aggregate validation only (needs restricted CSVs)
make reference   # refresh aggregate base-rates used by the app
```

Access notes: `docs/DATASETS.md`, `docs/SWAN_ACCESS.md`, `docs/MODEL_CARD.md`.

## 5. Adding a dataset

Follow `docs/ADDING_A_DATASET.md` (evidence adapter → rebuild foundation → tests).

## 6. What “reproducible” means here

| Claim | How we make it checkable |
|-------|---------------------------|
| Deterministic findings | Engine code + schema provenance; LLM never invents findings |
| No future leakage | `make audit` + causal mode in schema |
| Benchmark numbers | `make benchmark` regenerates `results/` |
| Model metrics | `results/model_*.json` + train scripts |
| No diagnostic claims | Safety guards + UI disclaimer |

## 7. Known honesty constraints

1. **Menopause-stage** may be synthetic until real SWAN is present — disclose in demos.
2. **mcPHASES** grounds aggregates only; we never ship participant rows.
3. The Aestra chat is a **demo UX** over the pipeline, not an autonomous medical agent.
4. **Benchmark cases** are synthetic (`honesty_note` in `results/benchmark_results.json`).

## 8. Artifact READMEs & contribution path

| Path | Purpose |
|------|---------|
| `models/README.md` | Checkpoints + honesty rules |
| `results/README.md` | Aggregate metrics inventory |
| `data/foundation/README.md` | Bundle license + rebuild |
| `CONTRIBUTING.md` | How to extend without breaking reproducibility |
| `docs/CHATGPT_SUBMISSION_BRIEF.md` | Pasteable explanation mapped to strong-submission criteria |
