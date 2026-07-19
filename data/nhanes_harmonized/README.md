# NHANES Harmonized Hormone Dataset (CycleBench export)

Derived from **NHANES 2017-March 2020 Pre-Pandemic** Sex Steroid Hormone Panel (P_TST)
and Demographics (P_DEMO), female participants, mapped into the CycleBench schema.

- Subjects: 6192
- Hormone measurements: 28484
- Reference-range rows: 48

## License
NHANES is US public domain (no restrictions). This harmonized export is released under
**CC-BY-4.0**. Cite NHANES (CDC/NCHS) and this project.

## IMPORTANT caveat
This is **cross-sectional** (one blood draw per person). It provides population reference
distributions and cross-sectional associations. It is **not** longitudinal and must **not**
be merged row-wise with mcPHASES participants, nor used to infer within-person cycle dynamics.

## Files
- `subjects.csv` — de-identified profiles (age band, sex)
- `hormone_events.csv` — CycleBench-typed hormone measurements with units + assay
- `reference_ranges.csv` — median / 2.5th / 97.5th percentile by analyte x age band
- `data_dictionary.csv` — NHANES variable -> analyte / unit / assay
