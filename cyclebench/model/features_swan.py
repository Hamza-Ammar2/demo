"""Menopause-stage features from SWAN public-use files (or synthetic fallback).

Target: coarse menopausal stage
  premenopausal | early_perimenopause | late_perimenopause | postmenopausal

Real SWAN (ICPSR) uses STATUS* / FSH* / E2* style variables that vary by visit.
This module:
  1. Loads `data/swan/` if present (CSV exports you place after ICPSR download)
  2. Otherwise builds a scientifically-plausible synthetic cohort for CI + offline demo
     (clearly labeled synthetic — never claimed as SWAN rows)

Association-only framing: stage *estimation* from hormones + symptoms + age, not a diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWAN_DIR = ROOT / "data" / "swan"

STAGES = [
    "premenopausal",
    "early_perimenopause",
    "late_perimenopause",
    "postmenopausal",
]

FEATURE_COLS = [
    "age_years",
    "fsh_miu_ml",
    "estradiol_pg_ml",
    "shbg_nmol_l",
    "hot_flash_freq",
    "night_sweat_freq",
    "sleep_disturbance",
    "cycle_irregularity",
    "amenorrhea_months",
    "bmi",
]


def _synthetic_swan(n: int = 2400, seed: int = 7) -> pd.DataFrame:
    """Generate a SWAN-like table with known stage → biomarker relationships."""
    rng = np.random.default_rng(seed)
    # stage priors roughly midlife cohort
    stages = rng.choice(
        STAGES, size=n, p=[0.28, 0.27, 0.22, 0.23]
    )
    rows = []
    for i, st in enumerate(stages):
        if st == "premenopausal":
            age = rng.normal(46, 3)
            fsh = rng.lognormal(mean=np.log(8), sigma=0.35)
            e2 = rng.lognormal(mean=np.log(80), sigma=0.45)
            hot = rng.poisson(0.4)
            night = rng.poisson(0.3)
            sleep = rng.integers(0, 2)
            irreg = rng.integers(0, 2)
            amen = 0
        elif st == "early_perimenopause":
            age = rng.normal(48, 3)
            fsh = rng.lognormal(mean=np.log(18), sigma=0.4)
            e2 = rng.lognormal(mean=np.log(70), sigma=0.55)
            hot = rng.poisson(1.5)
            night = rng.poisson(1.2)
            sleep = rng.integers(1, 3)
            irreg = rng.integers(1, 3)
            amen = rng.integers(0, 2)
        elif st == "late_perimenopause":
            age = rng.normal(51, 3)
            fsh = rng.lognormal(mean=np.log(40), sigma=0.4)
            e2 = rng.lognormal(mean=np.log(35), sigma=0.55)
            hot = rng.poisson(3.0)
            night = rng.poisson(2.5)
            sleep = rng.integers(2, 4)
            irreg = 2
            amen = rng.integers(2, 11)
        else:  # postmenopausal
            age = rng.normal(55, 4)
            fsh = rng.lognormal(mean=np.log(70), sigma=0.35)
            e2 = rng.lognormal(mean=np.log(15), sigma=0.4)
            hot = rng.poisson(2.2)
            night = rng.poisson(2.0)
            sleep = rng.integers(1, 4)
            irreg = 2
            amen = rng.integers(12, 48)

        shbg = rng.lognormal(mean=np.log(55), sigma=0.3)
        bmi = rng.normal(27, 5)
        rows.append({
            "participant_id": f"syn_{i}",
            "age_years": float(np.clip(age, 40, 65)),
            "fsh_miu_ml": float(np.clip(fsh, 1, 150)),
            "estradiol_pg_ml": float(np.clip(e2, 5, 400)),
            "shbg_nmol_l": float(np.clip(shbg, 10, 180)),
            "hot_flash_freq": float(min(hot, 10)),
            "night_sweat_freq": float(min(night, 10)),
            "sleep_disturbance": float(sleep),
            "cycle_irregularity": float(irreg),
            "amenorrhea_months": float(amen),
            "bmi": float(np.clip(bmi, 16, 50)),
            "menopause_stage": st,
            "data_source": "synthetic_swan_like",
        })
    return pd.DataFrame(rows)


def _load_real_swan(swan_dir: Path) -> pd.DataFrame | None:
    """Load a harmonized SWAN CSV if the user placed one.

    Expected columns (after your export/harmonize step):
      participant_id, age_years, fsh_miu_ml, estradiol_pg_ml, shbg_nmol_l,
      hot_flash_freq, night_sweat_freq, sleep_disturbance, cycle_irregularity,
      amenorrhea_months, bmi, menopause_stage
    """
    cand = swan_dir / "swan_harmonized.csv"
    if not cand.exists():
        # accept any single csv with required cols
        csvs = list(swan_dir.glob("*.csv"))
        if not csvs:
            return None
        cand = csvs[0]
    df = pd.read_csv(cand)
    need = {"menopause_stage", "age_years", "fsh_miu_ml"}
    if not need.issubset(df.columns):
        return None
    df = df.copy()
    df["data_source"] = "swan_icpsr"
    if "participant_id" not in df.columns:
        df["participant_id"] = [f"swan_{i}" for i in range(len(df))]
    # map common STATUS codes if present as ints
    stage_map = {
        1: "premenopausal", 2: "early_perimenopause",
        3: "late_perimenopause", 4: "postmenopausal",
        "1": "premenopausal", "2": "early_perimenopause",
        "3": "late_perimenopause", "4": "postmenopausal",
    }
    df["menopause_stage"] = df["menopause_stage"].replace(stage_map)
    df = df[df["menopause_stage"].isin(STAGES)]
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def build_menopause_table(swan_dir: Path | None = None, allow_synthetic: bool = True) -> pd.DataFrame:
    d = Path(swan_dir) if swan_dir else SWAN_DIR
    real = _load_real_swan(d) if d.exists() else None
    if real is not None and len(real) >= 50:
        return real
    if not allow_synthetic:
        raise FileNotFoundError(
            f"No usable SWAN CSV in {d}. See docs/SWAN_ACCESS.md"
        )
    return _synthetic_swan()


def matrix_from_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    y = df["menopause_stage"].to_numpy()
    pid = df["participant_id"].to_numpy()
    return X, y, pid, cols
