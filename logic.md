# logic.md — Core product & science logic (for ChatGPT / video scripts)

Use this file as the single brief when writing video narration, judge pitches, or
ChatGPT explanations. Prefer honesty over hype. Do not invent clinical claims.

---

## 1. One-sentence pitch

**Aestra** helps someone prepare for a doctor visit from chip-based symptoms and cycle
context; underneath, **CycleBench** is an open, leakage-aware stack for women’s
hormonal-health research (schema, engine, audit, benchmark, foundation, specialist models,
and a personalized phase model).

---

## 2. What problem we solve (and what we refuse)

Women’s hormonal symptoms are fragmented across apps, wearables, and labs. Generic chatbots
summarize text but invent certainty. We refuse diagnosis language. We produce:

- **Associations** (patterns that may be worth discussing)
- **Provenance** (what evidence or model produced a claim)
- **Honesty labels** (open vs restricted data; synthetic vs real metrics)

We do **not** claim: diagnosis, causation, hormone-level clinical assays, or “the world’s
first” foundation model.

---

## 3. Architecture (say this in the video)

```
User (Aestra chips / free text)
        │
        ├─ optional LLM: extract structured intake only (no medical conclusions)
        ▼
CycleBench Case (typed schema events)
        │
        ▼
Deterministic engine  →  timeline, cycle windows, patterns, confounders, missing info
        │
        ▼
Medical foundation graph  →  talking points + dataset evidence + ask-doctor questions
        │
        ├─ Model-pFL (phase GRU) + optional PCOS / menopause specialists
        ▼
Model-agent / LLM  →  warm doctor follow-up phrasing FROM model outputs + user words
        │                 (cannot invent a new phase or diagnosis)
        ▼
Soft doctor brief (UI): patterns → model signals → foundation → questions
```

**Invariant:** The LLM never computes a finding. Models and the engine do. The LLM only
structures language or rephrases already-computed text under a safety filter.

---

## 4. The three Hack-Nation tracks (how we map)

### Track 01 — Data foundations

**Goal:** Leave something researchers can use immediately.

**What we ship:**
- Open **NHANES** harmonized female hormone reference export (`data/nhanes_harmonized/`, CC-BY-4.0)
- **Dataset registry** with explicit “can support / must NOT” rules
- **CycleBench-Bench v0.1** — synthetic cases + transparent methodology (honest that cases are
  co-designed to test the engine, not clinical generalization)
- **Medical foundation** graph (seeded associations + evidence links)
- **Aggregate** mcPHASES validation only — raw PhysioNet data never redistributed

**What we do not ship:** A single downloadable multimodal longitudinal train/val/test pack
fusing wearables + labs + imaging for everyone. Restricted sources stay local under DUA.

**Video line:** “We publish the open pieces and the rules for the restricted ones — so future
work doesn’t pretend mcPHASES is free data.”

### Track 02 — AI model infrastructure

**Goal:** Focused, reproducible, explainable models — not a giant opaque net.

**Specialists (modular on purpose — not one fake mega-model):**
1. **Cycle phase / hormonal-state** — mcPHASES; sklearn baseline is weak (~0.28 balanced
   accuracy, barely above chance). We keep that number honest.
2. **Personalized FedPer GRU (Model-pFL)** — sequence model over 5-day windows; ~0.51
   balanced accuracy with real history; collapses toward chance with pad×5 (one day repeated).
3. **Menopause stage** — SWAN path when available; **shipped checkpoint metrics are often
   `synthetic_swan_like`** — say “illustrative until retrain on real SWAN.”
4. **PCOS-risk** — Kaggle clinical features; ~0.86 balanced accuracy; association / screening
   research signal, not a diagnosis. Follicle counts are near-diagnostic in that dataset —
   be careful not to oversell.

**Explainability:** Sklearn models expose feature importances. The GRU currently surfaces
input snapshots, not true attributions — say that clearly.

**Federated angle:** FedPer averages shared encoders across peer clients; local head stays
personal. Peers can come from Hugging Face when consented + configured, else local
mcPHASES simulation. We do **not** fabricate sync accuracy gains.

**Video line:** “We optimize for reproducibility and honesty over leaderboard theater.”

### Track 03 — Application infrastructure

**Goal:** One clear problem: appointment-ready soft read from self-report.

**Aestra flow:**
1. “I've been feeling off” chip conversation
2. Live soft read updates
3. **Model signals above medical foundation**
4. Questions to ask the doctor
5. Optional research consent (default off) for anonymized logs

**Model-pFL → Model-agent (core product logic Hamza wanted):**
- Model-pFL: intake → phase estimate
- Model-agent: phase estimate + user words → doctor follow-up language

**Video line:** “Better infrastructure shows up as a clearer conversation with your doctor —
not as a diagnosis in an app.”

---

## 5. Key components (file map for deep dives)

| Piece | Where |
|-------|--------|
| Schema / Case | `cyclebench/schema.py`, `cyclebench/case.py` |
| Engine | `cyclebench/engine/` |
| Leakage audit | `cyclebench/audit.py` |
| Benchmark | `cyclebench/benchmark/`, `results/benchmark_*.json` |
| Foundation | `cyclebench/foundation/`, `data/foundation/` |
| Intake mapping | `cyclebench/intake.py` |
| Sklearn models | `cyclebench/model/train.py`, `predict.py`, `tasks.py` |
| Model-pFL | `cyclebench/model/pfl.py` |
| LLM / Model-agent | `cyclebench/llm.py` (`extract_*`, `rephrase_opening`, `compose_doctor_followup`) |
| API | `api/main.py` (`/analyse/feeling-off`, `/models/*`) |
| UI | `web/` (Aestra) |
| Honest numbers | `docs/JUDGE_CARD.md`, `results/model_*.json` |
| Safety | `cyclebench/safety.py`, `docs/MEDICAL_SAFETY.md` |

---

## 6. Demo script (≈90 seconds)

1. Open home — brand **Aestra**.
2. Enter feeling-off; pick severe cramps + migraine; timing during period; last period recent.
3. Point to soft read: opening / patterns → **model signals** (phase) → foundation → ask doctor.
4. Say: “The phase estimate is Model-pFL. The words helping me talk to my doctor are
   Model-agent — they cannot invent a different phase.”
5. Close with disclaimer: associations for conversation, not diagnosis.
6. Optional: mention `make test` / `make benchmark` / open NHANES for researchers.

---

## 7. Numbers you may cite (and how)

| Claim | Safe phrasing |
|-------|----------------|
| PCOS ~0.86–0.90 | “Research association model on a public Kaggle clinical set; not a diagnostic device.” |
| Hormonal-state sklearn ~0.28 bal | “We publish a weak baseline honestly — chance is 0.25.” |
| pFL ~0.51 with W=5 | “Better with real multi-day history; sparse pad×5 is not magic.” |
| Menopause ~0.99 in JSON | **Only if** you say `synthetic_swan_like` / illustrative — otherwise don’t cite. |
| Engine vs naive benchmark | “Synthetic discrimination suite with honesty note — not clinical AUROC.” |
| mcPHASES engine agreement ~49% | Aggregate validation; raw data not shipped. |

When unsure, open `docs/JUDGE_CARD.md` and read `data_source` in the JSON.

---

## 8. Safety copy (must appear in videos)

- Does not diagnose, treat, or establish causation.
- Not a medical device.
- Restricted datasets stay behind DUA; we never commit raw mcPHASES/SWAN/Kaggle.
- Consented research upload is optional and off by default.

Forbidden vibes: “you have PCOS,” “you are in menopause,” “this causes your migraine,”
“clinically validated diagnostic AI.”

---

## 9. Talking points if judges push back

**“Isn’t phase prediction bad?”**  
Yes for sparse single-day intake — we show pad×5 honesty. With real sequences the GRU beats
the sklearn baseline. We don’t hide the weak numbers.

**“Why not one model on all datasets?”**  
Different populations and licenses. Modular specialists + a foundation graph is the honest
architecture. Fusing NHANES rows into mcPHASES subjects would be invalid.

**“Where’s the open multimodal dataset?”**  
NHANES open pack + registry + foundation + benchmark. Longitudinal multimodal gold stays
restricted; we document access and publish aggregates/adapters.

**“Is the LLM doing medicine?”**  
No. Extract + rephrase + follow-up phrasing only, safety-filtered, offline fallback.

---

## 10. Reproduce offline (say on camera)

```bash
make install
make test
make demo
make audit
make benchmark
make api
```

Optional: `make install-pfl && make pfl-smoke`.

---

## 11. Suggested video outline (3–5 min)

1. **Hook** — Sarah has ten minutes before the doctor; Aestra soft read.
2. **Track 03** — live demo of feeling-off (model signals → questions).
3. **Track 02** — cut to models: modular specialists, honest metrics table, pFL FedPer idea.
4. **Track 01** — NHANES open export, registry “must NOT,” mcPHASES stays local.
5. **Close** — clone, `make test`, extend; associations not diagnoses.

---

## 12. Glossary (keep consistent)

| Term | Meaning |
|------|---------|
| CycleBench | Scientific library / infrastructure |
| Aestra | Product UI / demo app |
| Model-pFL | Personalized FedPer GRU phase estimator |
| Model-agent | LLM (or offline template) that phrases doctor follow-ups |
| Soft read | Patient-facing brief: patterns, models, foundation, questions |
| Pad×5 | Repeat today’s features five times when history is short |
| FedPer | Federated personalization: shared encoder, local decision head |
| Foundation | Seeded medical association graph + evidence |

---

*End of brief. If something isn’t in this file or `docs/JUDGE_CARD.md`, don’t invent it —
check `results/*.json` or say “not claimed.”*
