# CycleBench Dataset Registry

## mcPHASES v1.0.0 (PhysioNet)
- License: PhysioNet Restricted Health Data License 1.5.0 (redistributable: False)
- Longitudinal: True | Subjects: 42 (20 with a 2nd interval)
- Modalities: wearable (Fitbit), CGM, urine hormone metabolites (Mira: LH/E3G/PdG), self-report symptoms, cycle phase labels
- **Can support:**
  - cycle-phase / hormonal-STATE classification (labeled)
  - symptom-vs-phase association analysis
  - wearable-vs-phase association analysis
  - within-person longitudinal pattern detection
  - leakage-audited, participant-split prediction baselines
- **Must NOT be used for:**
  - quantitative hormone-LEVEL prediction as a clinical output (noisy consumer proxy, n=42)
  - menopause-onset prediction (cohort is young reproductive-age; no peri/post-menopausal)
  - generalization beyond young, mostly-healthy menstruators without hormonal contraception
- Notes: Hormone values are Mira urine metabolites (semi-quantitative), manually entered.

## NHANES 2017-March 2020 (CDC)
- License: US public domain (redistributable: True)
- Longitudinal: False | Subjects: thousands (cross-sectional)
- Modalities: serum sex-steroid panel, demographics, body measures, biochemistry
- **Can support:**
  - population reference ranges (age/sex-stratified)
  - cross-sectional associations
  - harmonized open dataset publication
- **Must NOT be used for:**
  - within-person cycle dynamics (one blood draw per person)
  - row-wise merging with mcPHASES participants
  - longitudinal / temporal-leakage tasks
- Notes: Total testosterone is RDC-only for 2017+ (public only in 2013-2016 cycles).

## SWAN — Study of Women's Health Across the Nation (ICPSR public-use)
- License: ICPSR terms / study-specific; public-use files available (redistributable: False)
- Longitudinal: True | Subjects: ~3300 midlife women
- Modalities: serum hormones (E2, FSH, SHBG, …), vasomotor symptoms, sleep, menstrual/menopausal status
- **Can support:**
  - menopausal stage / transition classification
  - hormone + symptom trajectories across midlife
  - explainable multi-source menopause-stage models
- **Must NOT be used for:**
  - young-adult cycle-phase prediction (wrong age band)
  - ovarian cyst imaging diagnosis
  - exact calendar date of final menstrual period as a clinical device output
- Notes: Place harmonized CSV at data/swan/swan_harmonized.csv; see docs/SWAN_ACCESS.md.
