# Datasheet — NHANES Harmonized Hormone Dataset (CycleBench export)

Following Gebru et al., "Datasheets for Datasets."

## Motivation
Provide an open, reusable set of **age-stratified female sex-hormone reference distributions**
and CycleBench-typed measurements, so downstream tools can contextualize an individual's
values against a population — without redistributing restricted data.

## Composition
- **Source:** NHANES 2017–March 2020 Pre-Pandemic (CDC/NCHS): Sex Steroid Hormone Panel
  (P_TST) + Demographics (P_DEMO).
- **Instances:** 6,192 female participants; 28,484 hormone measurements.
- **Analytes:** estradiol (pg/mL), SHBG (nmol/L), FSH (mIU/mL), LH (mIU/mL), androstenedione
  (ng/dL), progesterone (ng/dL), each with assay/method recorded.
- **Files:** `subjects.csv`, `hormone_events.csv`, `reference_ranges.csv`,
  `data_dictionary.csv`, `README.md` in `data/nhanes_harmonized/`.
- **Reference ranges:** median / 2.5th / 97.5th percentile by analyte × decade age band
  (bands with n<20 dropped); 48 rows.

## Collection & processing
Derived deterministically by `cyclebench/adapters/nhanes_harmonize.py`: female filter
(RIAGENDR==2), analyte selection, NaN drop, decade age-banding, percentile computation.
No imputation. Reproducible via `make nhanes`.

## Uses
- **Appropriate:** population reference ranges, cross-sectional context, education/research.
- **Inappropriate:** within-person cycle dynamics, longitudinal/temporal tasks, row-wise
  merging with mcPHASES, individual diagnosis. Enforced by `adapters/registry.py`.

## Distribution & license
NHANES is US public domain; this harmonized export is released **CC-BY-4.0**. Cite CDC/NCHS
NHANES and this project (`CITATION.cff`).

## Maintenance
Versioned with the repo (schema v0.1.0). Regenerate with `make nhanes` after updating source
cycles. Known caveat: total testosterone is RDC-only for 2017+ and is therefore excluded.
