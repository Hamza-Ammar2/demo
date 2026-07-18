# Case Compiler / CycleBench — Build Specification (v1, revised)

> This is the revised build contract, tightened from the original prompt for a
> **one-day hackathon** and **grounded in the real datasets we already have**
> (`data/mcphases/`, `data/processed/` NHANES). Review this before build begins.
> Changes from the original prompt are flagged inline as **[CHANGED]** / **[ADDED]**.

---

## 0. Naming (unified) — [CHANGED]

Original had 3–4 overlapping names. Unify to two:

- **Case Compiler** — the user-facing *application* (intake → timeline → doctor brief).
- **CycleBench** — the open, reusable *scientific asset suite*: schema + deterministic
  engine + leakage/safety audit + benchmark + evaluation pipeline. The published DOI-able
  contribution. (No separate "CycleBench Engine" vs "Hormonal Timeline Benchmark" — the
  benchmark is **CycleBench-Bench v0.1**, the audit is **CycleBench-Audit**.)

---

## 1. One-sentence thesis

Turn a woman's fragmented longitudinal health history into an **inspectable, provenance-
linked case** for a doctor's appointment — where every finding is a *deterministic,
auditable* calculation, the LLM only translates language, and the whole engine + benchmark
is **validated on real mcPHASES data**, not just synthetic fixtures.

---

## 2. Guiding principles (non-negotiable)

1. **Deterministic core.** The LLM never computes a finding, date math, pattern, or
   confidence. It only: (a) extracts candidate events from language, (b) asks follow-ups,
   (c) explains validated findings in plain language, (d) renders the doctor brief from
   already-validated content.
2. **Association, never causation. Pattern, never diagnosis.** No disease prediction, no
   treatment advice, no risk scores. This is a rubric requirement, not a style choice.
3. **Every claim carries provenance.** No unsupported statement reaches the doctor brief.
4. **Offline-first demo.** — [ADDED] The headline Sarah flow MUST run with **zero network /
   no API key** (pre-extracted events). Live LLM is an enhancement, never a demo dependency.
5. **Grounded in real data.** — [ADDED] The engine's pattern detector, sleep confounder, and
   leakage audit are demonstrated on **real mcPHASES participants**, not only synthetic cases.
6. **License compliance (per-dataset).** — [ADDED] The challenge says publish datasets
   *"whenever possible"* AND *"comply with licensing."* Read together = publish what the license
   allows:
   - **mcPHASES = restricted (PhysioNet DUA).** **Never** commit mcPHASES rows or
     mcPHASES-derived *per-participant* data to the repo, fixtures, or `/results`. We publish the
     adapter code, pipeline, benchmark *definition*, **aggregate** metrics, and access
     instructions — not the raw data.
   - **NHANES = US public domain.** We **can and should** publish a cleaned/harmonized
     NHANES-hormone dataset mapped into the CycleBench schema as a real open artifact.
   - So `data/mcphases/` stays gitignored; a curated `data/nhanes_harmonized/` export IS
     publishable. Mandatory-regardless artifacts: benchmark + code + eval pipeline + docs
     (+ optional model checkpoint).

---

## 2b. Which layers we contribute + what we publish — [ADDED]

The challenge has 3 layers and says pick where you contribute most. We contribute:
- **Track 01 (Data & Benchmark) — primary.** CycleBench schema + CycleBench-Bench v0.1 +
  CycleBench-Audit + a harmonized open NHANES-hormone dataset.
- **Track 03 (Application) — the demo vehicle.** Case Compiler shows the infra in use.
- **Track 02 (Model) — optional/light.** One small reproducible baseline checkpoint (Tier D)
  so we can legitimately list "model checkpoint" among published artifacts.

**Framing rule:** lead the pitch and README with the *reusable asset* (CycleBench), not the
pretty app — otherwise we fall into the "isolated application" weak-submission trap.

**Open-science publish list (what actually ships under an open license):**
CycleBench schema (JSON Schema + Pydantic) · deterministic engine code · CycleBench-Audit ·
CycleBench-Bench v0.1 cases (synthetic) + runner + saved real metrics · harmonized NHANES
dataset (public domain) · mcPHASES adapter + setup instructions (NOT raw data) · aggregate
mcPHASES validation results · all docs · optional model checkpoint. Code = MIT, docs/data
artifacts = CC-BY-4.0.

---

## 3. Scope tiers — [CHANGED: this is the biggest change]

The original 16-item acceptance list is a full product. We build in **strict tiers** and
stop-loss by time. Do not advance a tier until the previous one is demoable.

### TIER A — Minimum Demoable Slice (target: first ~1/3 of time)
The thing we could submit if everything else fails.
1. Pydantic schema (core entities only: SubjectProfile, HealthEvent, SourceReference,
   CycleContext, Finding, DoctorBrief).
2. Deterministic engine: timeline compiler + cycle alignment (both modes) + pattern
   detector + sleep confounder + missing-info + provenance.
3. Pre-extracted **Sarah** fixture (no LLM needed) → produces a full DoctorBrief.
4. CLI or single script: `make demo` prints Sarah's doctor brief with provenance.
5. Core tests: pattern detector (pos/neg), confounder, retrospective-vs-causal, leakage
   rejection, provenance completeness, forbidden-language check, Sarah snapshot.

### TIER B — Grounding + Benchmark (target: middle ~1/3)
6. **Real mcPHASES validation** — [ADDED]: run the pattern detector + sleep confounder +
   cycle alignment on real participants who log `headaches`/`fatigue` in
   `hormones_and_selfreport.csv`; report how symptom episodes distribute across real `phase`
   labels, with sleep (`sleepissue` / `sleep.csv`) as confounder. Aggregate output only.
7. **CycleBench-Audit** leakage demo on a real prediction task: predict next-day symptom/phase
   from *past-only* wearable features, split **by participant**; show a leaking variant scores
   higher and the audit rejects it.
8. **CycleBench-Bench v0.1**: ~10 curated synthetic cases (not 25) — [CHANGED] — including ≥2
   negative, ≥1 misleading, ≥1 irregular-cycle, ≥1 insufficient-data. Runner + real metrics +
   machine- and human-readable outputs. Fixed seed.
9. Baselines: population rule + per-subject historical + phase-aware interpretable. **No ML
   model** unless Tier D reached. — [CHANGED]

### TIER C — API + Golden-Path Web (target: last ~1/3)
10. FastAPI: `/health`, `/cases/compile`, `/cases/{id}/doctor-brief`, `/cases/{id}/timeline`,
    `/benchmark/results`, `/audit/run`. In-memory storage. — [CHANGED: trimmed endpoint list]
11. Frontend: **2 pages** — [CHANGED] — Landing + Results (Doctor Mode + interactive timeline +
    finding detail as sections). Research/benchmark content can be a section on the results page
    or a third page only if time. Use a component library; tasteful, not a bespoke design system.
12. "Load Sarah demo" and "Compile My Case" both hit the real backend.

### TIER D — Stretch (only if all above done + tests green)
LLM live extraction, guided intake chat, editorial design system polish, ML/sequence baseline
+ **saved model checkpoint** (to legitimately claim a Track-02 artifact), PDF export, voice.
**Explicitly forbidden during core:** GRU, transformer, digital twin, disease classifier,
medication recommender.

---

## 4. Data schema (Phase 1)

Pydantic v2 models, versioned (`schema_version`). Entities as in the original prompt:
`SubjectProfile`, `HealthEvent`, `SourceReference`, `CycleContext`, `Finding`, `DoctorBrief`.
Event types as listed (symptom, menstrual_onset, estimated_cycle_phase, ovulation_proxy,
medication_started/stopped, dose_changed, contraception_changed, sleep_measurement,
stress_entry, wearable_measurement, glucose_measurement, hormone_measurement,
laboratory_result, clinical_encounter, free_text_note). Every event carries a
`certainty` and a `evidence_class` ∈ {patient_reported, documented, measured, inferred}.
Examples published in `docs/DATA_MODEL.md`. JSON Schema exported for reuse.

---

## 5. Deterministic engine (Phase 2)

As original, with emphasis:
- **Timeline compiler**: exact/approximate/range/conflicting/missing dates + certainty.
- **Cycle alignment**: `mode` is a **mandatory arg with no default**. `retrospective` may use
  the next known onset (label every output "retrospective"); `causal` uses only
  information available at the time. No assumed 28-day / day-14. Irregular cycles allowed but
  reduce confidence and are documented. Circular/phase-aware representation.
- **Pattern detector** (narrow golden path): episodes, episodes-in-window, baseline rate
  outside window, relative frequency / risk ratio when meaningful, repeated-cycle count,
  completeness, confidence category. **No diagnosis probability.**
- **Confounder detector**: poor sleep, medication/contraception changes, stress, missing/
  conflicting dates, incomplete logging. A finding is weakened when a confounder overlaps.
- **Missing-info detector**: highest-value missing fields (formulation, dose, missing period
  date, severity, missing baseline). No formal info-gain claims unless actually implemented.
- **Provenance engine**: every finding → supporting/contradicting/confounder event IDs +
  source excerpts + method + mode + limitations.

---

## 6. CycleBench-Audit (Phase 3)

Fails loudly. Assertions 1–10 from the original. — [CHANGED] Assertions about train/test
splits, train-only normalization, and no-target-leakage are exercised by the **real mcPHASES
prediction task** in Tier B (that's where they actually bite), plus a deliberately-leaking
fixture that the audit rejects. Demo line: *"A naïve model looked more accurate because it was
reading the future. Our audit caught it."*

---

## 7. Benchmark (Phase 4) — CycleBench-Bench v0.1

~10 documented synthetic cases (labeled synthetic), schema in `docs/BENCHMARK.md`. Metrics:
event extraction P/R, date accuracy, important-fact retention, pattern accuracy, false-pattern
rate, confounder recall, missing-info recall, provenance coverage, unsupported-claim count,
diagnosis/treatment-violation count, doctor-brief completeness, compression ratio. Two paths:
(A) generic LLM summary baseline, (B) Case Compiler pipeline. No API key → stored, clearly
labeled baseline fixtures. **Never fabricate results.** — [ADDED] Honesty note in docs: LLM-
authored "messy" narratives ≠ real patient language; a few hand-written messy narratives
included to mitigate round-trip circularity.

---

## 8. Dataset adapters (Phase 7)

- **mcPHASES adapter** — real, used in Tier B. Inspect actual columns (already done:
  `id, study_interval, day_in_study, phase, lh, estrogen, pdg, headaches, cramps, fatigue,
  sleepissue, moodswing, stress, ...`; wearables join on `id`/`day_in_study`). No invented
  columns. Aggregate outputs only; raw data never committed.
- **NHANES adapter + open dataset** — [CHANGED] cross-sectional only (demographics,
  reproductive, thyroid/hormone labs, nutrition, meds). Never merged row-wise with mcPHASES.
  Because NHANES is **public domain**, also emit a **harmonized, openly-publishable**
  `data/nhanes_harmonized/` export mapped into the CycleBench schema (with units + assay
  provenance + reference-range stratification) — this is our legally-shippable "publish an open
  dataset" deliverable. Both sources map into the shared schema while retaining cohort identity,
  assay provenance, units, collection context, and longitudinal-vs-cross-sectional status.
- **Dataset registry**: what each source can/can't validly support.

---

## 9. LLM integration (Phase 5)

Provider abstraction: `extract_events`, `generate_follow_up_questions`, `explain_finding`,
`generate_doctor_brief`. Env-configured model name, no hardcoded keys, Pydantic-validated JSON,
retry-once-then-safe-fallback, never pass invalid output into the engine. Deterministic demo
fallback mandatory (see Principle 4).

---

## 10. Repo layout

```
/web        Next.js + TS + Tailwind + a component lib (2 pages)
/api        FastAPI + Pydantic + pandas/numpy/scipy/sklearn + pytest
/cyclebench engine + audit + benchmark + adapters (the reusable core, importable)
/data       gitignored; adapters + setup instructions only
/docs       DATA_MODEL, BENCHMARK, DATASETS, LIMITATIONS, MEDICAL_SAFETY, ARCHITECTURE,
            DATASHEET (added), DEMO (added: video storyboard + 3-min script), DECISIONS
/results    aggregate benchmark outputs only (no per-participant real data)
root        README, LICENSE (MIT code / CC-BY-4.0 docs), CITATION.cff, .env.example,
            Makefile, requirements pinned, TASKS.md, SPEC.md
```
Commands: `make install | test | demo | benchmark | audit | dev`.

---

## 11. Medical safety + docs

Safety language rules and the standard disclaimer everywhere (verbatim from original §MEDICAL
SAFETY). `docs/LIMITATIONS.md` mandatory and honest (no diagnosis/treatment/validation; small
open-dataset + synthetic-benchmark limits; retrospective-vs-causal; irregular/missing cycle
data; population bias; assay/unit provenance; LLM-extraction limits). — [ADDED] `docs/DATASHEET.md`
(Datasheets-for-Datasets style) for CycleBench-Bench. Pin dependency versions for reproducibility.

---

## 12. Acceptance = Tier A + Tier B green

MVP is "done" when Tier A and Tier B pass (deterministic Sarah brief with provenance,
retrospective/causal visible, leakage fixture rejected, benchmark runs with real saved metrics,
real-mcPHASES validation produces aggregate results, core tests green). Tier C/D are upside.

---

## 13. Build behavior

Inspect repo → write `TASKS.md` (gap analysis + tiered checklist) → build Tier A fully →
tests alongside each module → advance tiers without asking permission → keep runnable after
each step → record assumptions in `docs/DECISIONS.md` → never fake results or claim
inaccessible data was integrated → prefer a smaller working slice to broad half-work. Report
after each tier: files changed, commands, tests passing, blockers, next tier.
