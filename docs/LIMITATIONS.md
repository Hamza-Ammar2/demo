# Limitations

## Not a medical device
Case Compiler organizes information and detects **temporal associations**. It does not
diagnose, does not recommend treatment, and does not establish causation. Output is intended
to help a patient and clinician have a better conversation.

## What we deliberately do NOT do
- **No hormone-level prediction as a clinical output.** mcPHASES hormones are semi-quantitative
  consumer urine metabolites (Mira), manually entered, n=42. NHANES is a single cross-sectional
  serum draw. Neither supports trustworthy within-person hormone-value forecasting.
- **No menopause / peri-menopause onset prediction.** The longitudinal cohort is young and
  reproductive-age; there is no peri/post-menopausal longitudinal signal to learn from.
- **No causal claims.** Confounders (sleep, stress, medication changes) are surfaced, not
  adjusted away.

## Methodological limits
- **Cycle alignment** uses bleeding-onset spacing, not measured hormones; agreement with
  hormone-based labels on mcPHASES is ~49% (still far above the 25% four-class chance rate).
- **Causal mode** is intentionally conservative and will report fewer aligned episodes than
  retrospective mode; this is a feature (no future-information leakage), not a bug.
- **Benchmark** cases are synthetic and co-designed with the engine's thresholds; they test
  discrimination and safety, not generalization (see BENCHMARK.md).
- Small n and self-report bias throughout; findings are hypotheses for a clinician, not facts.

## Population / equity
Cohorts skew toward specific demographics and mostly-healthy menstruators not on hormonal
contraception. Findings may not generalize across ages, conditions (e.g. PCOS, endometriosis),
or contraceptive states. This is stated in-product, not hidden.
