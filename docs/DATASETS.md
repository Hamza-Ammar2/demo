# Datasets

See `docs/DATASET_REGISTRY.md` for the machine-generated capability matrix.

## mcPHASES (PhysioNet) — RESTRICTED, not redistributed

Longitudinal multimodal menstrual dataset (42 participants; Fitbit, CGM, daily Mira urine
hormone metabolites LH/E3G/PdG, self-reported symptoms, cycle-phase labels).

**License:** PhysioNet Restricted Health Data License 1.5.0. **We do not and cannot
redistribute it.** This repo contains only adapter code and **aggregate** statistics.

### How to obtain it
1. Create a PhysioNet account and complete any required training.
2. Sign the data use agreement at https://physionet.org/content/mcphases/1.0.0/
3. Download and place the CSVs under `data/mcphases/` (gitignored).
4. Run `make mcphases` → writes aggregate results to `results/mcphases_validation.json`.

### What we use it for (aggregate only)
- **Symptom-vs-phase clustering:** headache episodes cluster significantly by cycle phase
  (χ²=18.9, p=0.0003), peaking in the menstrual phase; fatigue peaks menstrually but is not
  significant (p=0.12) — reported honestly.
- **Engine validation:** CycleBench's cycle-alignment (phase derived from bleeding onsets
  alone) agrees with mcPHASES' hormone-based Mira labels ~49.5% of days (vs 25% chance, 4
  classes).
- **Sleep confounder:** ~38% of headache-episode days also report high sleep issues.

### What we do NOT use it for
Quantitative hormone-level prediction (noisy consumer proxy, n=42) or menopause prediction
(cohort is young reproductive-age). See LIMITATIONS.

## NHANES (CDC) — PUBLIC DOMAIN, harmonized export published

Cross-sectional US survey. We use the 2017–March 2020 Sex Steroid Hormone Panel (P_TST) +
demographics (P_DEMO).

- `make nhanes` builds `data/nhanes_harmonized/` (CC-BY-4.0): 6,192 female subjects, 28k
  hormone measurements, 48 age-stratified reference ranges (estradiol, SHBG, FSH, LH,
  androstenedione, progesterone) with units + assay provenance.
- **Cross-sectional only** — one blood draw per person; never merged row-wise with mcPHASES,
  never used for within-person cycle dynamics.
- Note: total testosterone is RDC-only for 2017+ (public only in 2013–2016 cycles).
