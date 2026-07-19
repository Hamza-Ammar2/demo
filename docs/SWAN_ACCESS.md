# Accessing SWAN for the menopause-stage model

The menopause-stage model (`menopause_stage_v0.1`) is designed for **SWAN**
(Study of Women's Health Across the Nation) public-use data on ICPSR.

## Why SWAN
SWAN is the dataset that actually supports the challenge bullet on biological
signatures of the menopausal transition (hormones + symptoms + stage labels over time).
mcPHASES cannot: it is a young reproductive-age cohort.

## Offline / CI behavior
Until real SWAN files are present, `make train-models` trains on a **synthetic
SWAN-like cohort** (clearly labeled `data_source=synthetic_swan_like`). That keeps
demos and tests green. Publication metrics should be re-run on real ICPSR data.

## How to get real SWAN (public-use)
1. Create a free ICPSR account: https://www.icpsr.umich.edu/
2. Download SWAN Baseline (ICPSR 28762) and later visits as needed:
   https://www.swanstudy.org/swan-research/data-access/
3. Export/harmonize into a single CSV at:
   `data/swan/swan_harmonized.csv`
4. Required columns:
   - `participant_id`
   - `age_years`
   - `fsh_miu_ml`
   - `estradiol_pg_ml`
   - `shbg_nmol_l` (optional but used)
   - `hot_flash_freq`, `night_sweat_freq`, `sleep_disturbance`
   - `cycle_irregularity`, `amenorrhea_months`, `bmi`
   - `menopause_stage` ∈ {premenopausal, early_perimenopause, late_perimenopause, postmenopausal}
     (integer codes 1–4 are also accepted)
5. Re-run: `make train-models`

## Framing (medical safety)
The model estimates a **stage category** from hormones/symptoms/age.
It does **not** diagnose menopause or predict an exact onset date.
Outputs are wrapped as `possible` Findings with association-only language.
