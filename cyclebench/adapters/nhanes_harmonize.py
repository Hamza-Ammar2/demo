"""NHANES harmonizer -> publishable open dataset.

NHANES is US public domain, so unlike mcPHASES we CAN redistribute a derived,
harmonized export. This maps the 2017-March 2020 sex-steroid panel (P_TST) + demographics
(P_DEMO) for female participants into the CycleBench schema, preserving units and assay
provenance, and emits age-stratified reference ranges.

Outputs (to data/nhanes_harmonized/, which IS committed as the open artifact):
  subjects.csv          - de-identified subject profiles (SEQN, age band, sex)
  hormone_events.csv     - one row per hormone measurement, CycleBench-typed
  reference_ranges.csv   - median / 2.5th / 97.5th percentile by analyte x age band
  data_dictionary.csv    - variable -> analyte, unit, assay
  README.md              - provenance, license, and "cross-sectional only" caveat
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "nhanes_harmonized"

# NHANES variable -> (analyte name, unit, assay/method).
ANALYTES = {
    "LBXEST": ("estradiol", "pg/mL", "ID-LC-MS/MS"),
    "LBXSHBG": ("shbg", "nmol/L", "immunoassay (chemiluminescence)"),
    "LBXFSH": ("fsh", "mIU/mL", "immunoassay"),
    "LBXLUH": ("lh", "mIU/mL", "immunoassay"),
    "LBXAND": ("androstenedione", "ng/dL", "ID-LC-MS/MS"),
    "LBXPG4": ("progesterone", "ng/dL", "ID-LC-MS/MS"),
}


def _age_band(age) -> str:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if a < 18:
        return "under_18"
    for lo in range(18, 80, 10):
        if lo <= a < lo + 10:
            return f"{lo}-{lo + 9}"
    return "80_plus"


def harmonize() -> dict:
    import numpy as np
    import pandas as pd

    demo_p = PROCESSED / "P_DEMO.csv"
    tst_p = PROCESSED / "P_TST.csv"
    if not (demo_p.exists() and tst_p.exists()):
        raise FileNotFoundError(
            f"NHANES processed CSVs not found in {PROCESSED}. "
            "Run the NHANES download/convert step first (see docs/DATASETS.md)."
        )

    demo = pd.read_csv(demo_p, usecols=["SEQN", "RIAGENDR", "RIDAGEYR"])
    tst = pd.read_csv(tst_p, usecols=["SEQN"] + [c for c in ANALYTES if c in pd.read_csv(tst_p, nrows=0).columns])

    # Female participants only (RIAGENDR == 2).
    demo = demo[demo["RIAGENDR"] == 2].copy()
    df = demo.merge(tst, on="SEQN", how="inner")

    OUT.mkdir(parents=True, exist_ok=True)

    # --- subjects.csv ---
    subjects = pd.DataFrame({
        "subject_id": "nhanes_" + df["SEQN"].astype(int).astype(str),
        "sex": "female",
        "age_years": df["RIDAGEYR"],
        "age_band": df["RIDAGEYR"].map(_age_band),
        "longitudinal": False,
        "cohort": "NHANES 2017-Mar2020 (P)",
    })
    subjects.to_csv(OUT / "subjects.csv", index=False)

    # --- hormone_events.csv (CycleBench-typed, cross-sectional) ---
    rows = []
    for var, (analyte, unit, assay) in ANALYTES.items():
        if var not in df.columns:
            continue
        sub = df[["SEQN", "RIDAGEYR", var]].dropna(subset=[var])
        for _, r in sub.iterrows():
            rows.append({
                "subject_id": f"nhanes_{int(r['SEQN'])}",
                "event_type": "hormone_measurement",
                "analyte": analyte,
                "value": float(r[var]),
                "unit": unit,
                "assay": assay,
                "evidence_class": "measured",
                "cross_sectional": True,
                "age_band": _age_band(r["RIDAGEYR"]),
            })
    events = pd.DataFrame(rows)
    events.to_csv(OUT / "hormone_events.csv", index=False)

    # --- reference_ranges.csv ---
    ref_rows = []
    for var, (analyte, unit, assay) in ANALYTES.items():
        if var not in df.columns:
            continue
        for band, g in df.dropna(subset=[var]).groupby(df["RIDAGEYR"].map(_age_band)):
            vals = g[var].to_numpy()
            if len(vals) < 20:
                continue
            ref_rows.append({
                "analyte": analyte, "unit": unit, "sex": "female", "age_band": band,
                "n": int(len(vals)),
                "median": round(float(np.median(vals)), 3),
                "p2_5": round(float(np.percentile(vals, 2.5)), 3),
                "p97_5": round(float(np.percentile(vals, 97.5)), 3),
            })
    ref = pd.DataFrame(ref_rows).sort_values(["analyte", "age_band"])
    ref.to_csv(OUT / "reference_ranges.csv", index=False)

    # --- data_dictionary.csv ---
    dd = pd.DataFrame(
        [{"nhanes_variable": v, "analyte": a, "unit": u, "assay": assay}
         for v, (a, u, assay) in ANALYTES.items()]
    )
    dd.to_csv(OUT / "data_dictionary.csv", index=False)

    _write_readme(len(subjects), len(events), len(ref))

    return {
        "n_subjects": int(len(subjects)),
        "n_hormone_events": int(len(events)),
        "n_reference_ranges": int(len(ref)),
        "analytes": sorted({a for _, (a, _, _) in ANALYTES.items()}),
        "output_dir": str(OUT.relative_to(ROOT)),
    }


def _write_readme(n_subj, n_ev, n_ref) -> None:
    (OUT / "README.md").write_text(f"""# NHANES Harmonized Hormone Dataset (CycleBench export)

Derived from **NHANES 2017-March 2020 Pre-Pandemic** Sex Steroid Hormone Panel (P_TST)
and Demographics (P_DEMO), female participants, mapped into the CycleBench schema.

- Subjects: {n_subj}
- Hormone measurements: {n_ev}
- Reference-range rows: {n_ref}

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
""")


def main() -> int:
    import json
    out = harmonize()
    print(json.dumps(out, indent=2))
    print(f"\nHarmonized open dataset written to {out['output_dir']} (CC-BY-4.0, publishable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
