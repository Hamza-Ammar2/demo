# ChatGPT submission brief — Aestra / CycleBench

Copy everything below the line into ChatGPT (or any writing assistant) when you need
help drafting a pitch, README blurb, demo script, or judge Q&A. It is written so the
model maps our repo onto the hackathon “strong vs weak submission” chart.

---

## Prompt (paste from here)

You are helping me write clear, honest materials for a hackathon submission:
**Hack-Nation — Open AI Infrastructure for Women’s Hormonal Health.**

Judge the language against this rubric (strong vs weak). Prefer **strong** framing;
never invent assets we do not have; never make diagnostic claims.

### Rubric (what judges reward)

**1. Openness and reusability**
- Strong: Publish reusable datasets, benchmarks, model checkpoints, and evaluation
  pipelines under an open license.
- Weak: Build an isolated application with no reusable dataset, benchmark, or model
  contribution.

**2. Problem definition and transparency**
- Strong: Solve one clearly defined prediction or infrastructure problem exceptionally
  well, with transparent methods and evidence.
- Weak: Depend on undocumented proprietary data or hide core assumptions, preprocessing,
  and evaluation choices.

**3. Reproducibility and scientific rigor**
- Strong: Share reproducible code and documentation so the research community can extend
  the work after the hackathon.
- Weak: Make unsupported medical or diagnostic claims, or present a polished interface
  without scientific validation.

### What we built (names)

- **Aestra** = the user-facing demo app (Track 03): chip-based intake → soft read for a
  doctor visit. Static `web/` served by FastAPI. Not a chatbot that invents medicine.
- **CycleBench** = the open scientific stack underneath (Tracks 01 + 02): schema,
  deterministic engine, leakage audit, synthetic benchmark, medical foundation graph,
  specialist models, adapters.

Pitch rule: We are **not** claiming we *are* “the world’s first” open AI infrastructure
alone. We claim we contributed **reusable open building blocks** others can extend.

### Architecture (how code works)

```
User chips / optional free text
        │
        ▼
[optional LLM]  ← ears/mouth only: extract language into schema fields
        │         NEVER computes a Finding
        ▼
CycleBench Case (Pydantic schema + JSON Schema in docs/schema/)
        │
   ┌────┼────────────────────┐
   ▼    ▼                    ▼
engine  foundation graph   specialist models
(timeline, cycle align,    (prob + feature
 patterns, confounders,     attributions only)
 missing info)
   │    │                    │
   └────┼────────────────────┘
        ▼
DoctorBrief / soft read + CycleBench-Audit (leakage/safety)
        ▼
Aestra UI results takeover
```

**Invariant:** every `Finding` is produced by deterministic, inspectable Python and
carries provenance. The LLM never invents clinical edges or pattern claims.

### Map repo → strong-submission columns

| Rubric column | What we ship | Where in the repo |
|---------------|--------------|-------------------|
| Reusable dataset / export | Harmonized NHANES female hormone export (CC-BY-4.0) | `data/nhanes_harmonized/` |
| Reusable schema | Longitudinal hormonal-health case schema | `cyclebench/schema.py`, `docs/schema/` |
| Benchmark + eval pipeline | CycleBench-Bench v0.1 (Path A naive vs Path B CycleBench) | `cyclebench/benchmark/`, `make benchmark`, `results/benchmark_results.json` |
| Leakage / safety science | CycleBench-Audit (future-info + unsafe claim checks) | `cyclebench/audit.py`, `make audit` |
| Medical knowledge substrate | Versioned foundation graph (entities, associations, evidence) | `cyclebench/foundation/`, `data/foundation/foundation_v0.1.json` |
| Model checkpoints + cards | hormonal-state, menopause-stage, PCOS-risk | `models/*.joblib`, `results/model_*.json`, `docs/MODEL_CARD.md` |
| Reproducible entrypoints | Makefile targets, tests, docs | `make install/test/demo/audit/benchmark/api`, `docs/REPRODUCIBILITY.md`, `CONTRIBUTING.md` |
| Application that *uses* infra | Aestra demo (does not replace the science) | `web/`, `api/` |

### Honest data / model labels (do not hide these)

| Asset | Source | Redistribute raw? | Honesty label |
|-------|--------|-------------------|---------------|
| Hormonal-state model | PhysioNet **mcPHASES** | **No** (DUA) | Checkpoint/metrics may ship; `data_source=mcphases` |
| PCOS-risk model | **Kaggle PCOS** clinic cohort | **No** (re-download) | `data_source=kaggle_pcos` |
| Menopause-stage model | Real SWAN if present, else **synthetic_swan_like** | Synthetic yes / SWAN no | Always read `results/model_menopause_stage.json` → `data_source`. ~99% accuracy on synthetic is **illustrative**, not clinical validation |
| NHANES harmonized | CDC public | **Yes** | Open export |
| Benchmark cases | Synthetic, co-designed with thresholds | Yes | Tests discrimination + safety, **not** clinical generalization (`honesty_note` in results JSON) |
| Foundation evidence | Mix of guideline seed + open aggregates + local adapters | Bundle yes; restricted rows never | Restricted sources = aggregates/model signals only |

### What “AI” means in this project (stay precise)

- Literally trained ML: the three specialist predictors (joblib).
- Infrastructure *around* AI: schema, deterministic engine, audit, foundation graph,
  evaluation harness — these are the reusable scientific contribution, not “an agent.”
- Chat UI: demo UX over a **fixed pipeline**, not an omniscient medical agent.

### Safety / claims we never make

- Not a medical device; not a diagnosis; not treatment advice.
- No clinical menopause **onset timing** prediction claim.
- No unsupported diagnostic language in briefs (safety guards).
- Soft read = talking points + questions for a clinician, with confounders and missing info.

### Offline reproduction judges can run (no keys, no restricted data)

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make test          # ~78 tests
make demo          # Sarah doctor brief
make audit         # leakage fixture rejected
make benchmark     # writes results/benchmark_*.json
make api           # http://127.0.0.1:8000 → Aestra UI
```

Optional rebuilds needing local data: `make train-models`, `make train-tasks`,
`make mcphases`, `make foundation` — documented in `docs/REPRODUCIBILITY.md`.

### License

- Code: MIT
- Docs + foundation packaging + NHANES harmonized export: CC-BY-4.0
- Cite: `CITATION.cff`

### Writing instructions for you (ChatGPT)

1. Lead with **reusable open artifacts** (schema, benchmark, audit, foundation, models,
   NHANES), then mention Aestra as the **demo application** that consumes them.
2. For each claim, name a file, Makefile target, or metric JSON so a judge can verify.
3. Explicitly disclose synthetic menopause and synthetic benchmark cases; never pad
   accuracy as clinical proof.
4. Frame success as “others can extend after the hackathon” (adapters, foundation
   evidence, new model tasks, new bench cases) — see `CONTRIBUTING.md` /
   `docs/ADDING_A_DATASET.md`.
5. Do **not** say we are the world’s first infrastructure; say we contribute layers
   toward that shared mission.
6. Keep medical tone association / counseling only.
7. Prefer short, concrete language over hype.

When I ask you for a pitch, README section, demo script, or FAQ answer, rewrite it
using the facts above and the strong-submission column of the rubric.
