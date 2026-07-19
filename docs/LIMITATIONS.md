# Limitations

## Not a medical device
**Aestra** (application) and **CycleBench** (scientific stack) organize information and
detect **temporal associations**. They do not diagnose, recommend treatment, or establish
causation. Output is intended to help a patient and clinician have a better conversation.

## What we deliberately do NOT do
- **No hormone-level prediction as a clinical output.** mcPHASES hormones are semi-quantitative
  consumer urine metabolites (Mira), manually entered, small n. NHANES is a single
  cross-sectional serum draw. Neither supports trustworthy within-person hormone-value forecasting.
- **No clinical menopause *onset timing* prediction.** mcPHASES is a young reproductive-age
  cohort and cannot support peri/post onset modeling by itself.
  - Separately, CycleBench ships a **menopause-stage *category*** model (research/demo signal).
    When real SWAN is absent it trains on a **synthetic SWAN-like** cohort and must be
    disclosed via `data_source=synthetic_swan_like` in `results/model_menopause_stage.json`.
    High accuracy on synthetic data is **illustrative**, not clinical validation.
- **No causal claims.** Confounders (sleep, stress, medication changes) are surfaced, not
  adjusted away.
- **No unsupported diagnostic claims in UI copy.** Safety guards block affirmative diagnosis /
  treatment language in briefs.

## Methodological limits
- **Cycle alignment** uses bleeding-onset spacing, not measured hormones; agreement with
  hormone-based labels on mcPHASES is ~49% (still far above the 25% four-class chance rate).
- **Causal mode** is intentionally conservative and will report fewer aligned episodes than
  retrospective mode; this is a feature (no future-information leakage), not a bug.
- **Benchmark** cases are synthetic and co-designed with the engine's thresholds; they test
  discrimination and safety, not clinical generalization (see `docs/BENCHMARK.md`).
- Small n and self-report bias throughout; findings are hypotheses for a clinician, not facts.

## Population / equity
Cohorts skew toward specific demographics. PCOS-risk was trained on a Kerala clinic Kaggle
cohort; hormonal-state on mcPHASES (n≈42 participants). Findings may not generalize across
ages, conditions, or contraceptive states. This is stated in docs and product disclaimers.
