"""Corpus reference statistics — the shippable, aggregate face of the database.

The raw corpus (mcPHASES restricted, SWAN, NHANES) never leaves the machine. What
the *app* is allowed to use is this aggregate, non-identifying summary:

  * symptom -> menstrual-phase base rates (from real mcPHASES participant-days)
  * menopause-stage feature priorities (from the trained SWAN-stage model)
  * NHANES analyte reference ranges (age-banded)

This keeps us license-clean (no per-participant rows) while letting the chat ground
its read in real cohort statistics rather than hand-waving.

Build once from the local corpus (`python -m cyclebench.reference`); the resulting
`results/reference_stats.json` is committed and loaded at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
MCPHASES = ROOT / "data" / "mcphases" / "hormones_and_selfreport.csv"
NHANES_RANGES = ROOT / "data" / "nhanes_harmonized" / "reference_ranges.csv"
MENO_META = ROOT / "models" / "menopause_stage_v0.1.meta.json"
OUT = ROOT / "results" / "reference_stats.json"

ORDINAL = {
    "Not at all": 0, "Very Low/Little": 1, "Low": 2,
    "Moderate": 3, "High": 4, "Very High": 5,
}
EPISODE_THRESHOLD = 4
PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
SYMPTOM_COLS = [
    "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
    "moodswing", "stress", "foodcravings", "bloating",
]

# intake symptom type -> mcPHASES self-report column
INTAKE_TO_MCPHASES = {
    "headache": "headaches",
    "migraine": "headaches",
    "cramps": "cramps",
    "pelvic_pain": "cramps",
    "fatigue": "fatigue",
    "brain_fog": "fatigue",
    "mood": "moodswing",
    "bloating": "bloating",
    "sore_breasts": "sorebreasts",
}

_CACHE: Optional[dict] = None


def _to_num(v):
    if v in ORDINAL:
        return ORDINAL[v]
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def build_reference() -> dict:
    """Compute the aggregate reference from the local corpus."""
    ref: dict = {"sources": {}, "symptom_phase": {}, "menopause": {}, "nhanes": {}}

    # ---- mcPHASES: symptom -> phase episode rates (aggregate over participant-days) ----
    if MCPHASES.exists():
        import pandas as pd
        from scipy.stats import chisquare

        df = pd.read_csv(MCPHASES)
        df = df[df["phase"].isin(PHASES)].copy()
        ref["sources"]["mcphases"] = {
            "n_participants": int(df["id"].nunique()),
            "n_participant_days": int(len(df)),
            "license": "PhysioNet restricted — aggregate stats only, no rows shipped",
        }
        for sym in SYMPTOM_COLS:
            if sym not in df.columns:
                continue
            num = df[sym].map(_to_num)
            valid = num.notna()
            if valid.sum() < 100:
                continue
            sub = df[valid].copy()
            epi = (num[valid] >= EPISODE_THRESHOLD).astype(int)
            days_per_phase = sub.groupby("phase").size()
            epi_per_phase = sub.assign(_e=epi.values).groupby("phase")["_e"].sum()
            rate = (epi_per_phase / days_per_phase).round(4)
            total_epi = int(epi_per_phase.sum())
            phases = [p for p in PHASES if p in epi_per_phase.index]
            observed = [int(epi_per_phase[p]) for p in phases]
            expected = [(days_per_phase[p] / days_per_phase.sum()) * total_epi for p in phases]
            chi2 = pval = None
            if total_epi >= 20 and all(e > 0 for e in expected):
                chi2, pval = chisquare(f_obs=observed, f_exp=expected)
            ref["symptom_phase"][sym] = {
                "n_episodes": total_epi,
                "episode_rate_by_phase": rate.to_dict(),
                "dominant_phase": rate.idxmax() if len(rate) else None,
                "chi2": round(float(chi2), 2) if chi2 is not None else None,
                "p_value": round(float(pval), 5) if pval is not None else None,
                "significant": bool(pval is not None and pval < 0.05),
            }

    # ---- SWAN-stage model: which features actually move the prediction ----
    if MENO_META.exists():
        meta = json.loads(MENO_META.read_text())
        ref["menopause"] = {
            "classes": meta.get("classes", []),
            "feature_importances": (meta.get("metrics", {}) or {}).get("feature_importances", []),
            "data_source": (meta.get("metrics", {}) or {}).get("data_source", "unknown"),
            "balanced_accuracy": (meta.get("metrics", {}) or {}).get("balanced_accuracy"),
            "note": "Feature priorities from the trained menopause-stage model.",
        }

    # ---- NHANES: analyte reference ranges (age-banded), already aggregate/public ----
    if NHANES_RANGES.exists():
        import pandas as pd
        r = pd.read_csv(NHANES_RANGES)
        ref["nhanes"]["reference_ranges"] = r.to_dict(orient="records")
        ref["sources"]["nhanes"] = {"license": "public domain (CDC/NHANES)"}

    return ref


def load_reference(rebuild: bool = False) -> dict:
    """Load the cached aggregate reference (build if missing and corpus is present)."""
    global _CACHE
    if _CACHE is not None and not rebuild:
        return _CACHE
    if OUT.exists() and not rebuild:
        _CACHE = json.loads(OUT.read_text())
        return _CACHE
    ref = build_reference()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=2))
    _CACHE = ref
    return ref


def symptom_cohort_context(intake_type: str) -> Optional[dict]:
    """Return an aggregate cohort context line for an intake symptom, or None.

    Never returns per-participant data — only the shipped aggregate summary.
    """
    ref = load_reference()
    col = INTAKE_TO_MCPHASES.get(intake_type)
    if not col:
        return None
    stat = ref.get("symptom_phase", {}).get(col)
    if not stat or not stat.get("dominant_phase"):
        return None
    src = ref.get("sources", {}).get("mcphases", {})
    n = src.get("n_participants")
    phase = stat["dominant_phase"]
    p = stat.get("p_value")
    if stat.get("significant"):
        sentence = (
            f"In the mcPHASES cohort (n={n}), reported {col} clustered most in the "
            f"{phase.lower()} phase (chi\u00b2={stat['chi2']}, p={p})."
        )
    else:
        sentence = (
            f"In the mcPHASES cohort (n={n}), reported {col} did not cluster "
            f"significantly by cycle phase (p={p})."
        )
    return {
        "symptom": intake_type,
        "mcphases_column": col,
        "dominant_phase": phase,
        "p_value": p,
        "significant": stat.get("significant", False),
        "sentence": sentence,
    }


def main() -> int:
    ref = build_reference()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ref, indent=2))
    n_sym = len(ref.get("symptom_phase", {}))
    print(f"Reference built: {n_sym} symptom-phase profiles, "
          f"{len(ref.get('menopause', {}).get('feature_importances', []))} menopause features.")
    print(f"Written to {OUT.relative_to(ROOT)} (aggregate only — no raw rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
