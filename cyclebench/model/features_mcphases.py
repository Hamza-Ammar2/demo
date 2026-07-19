"""Multi-source feature builder for hormonal-state prediction (mcPHASES).

Integrates: self-report symptoms, resting HR, sleep, steps, stress, wrist temp, CGM.
Target: Mira cycle phase label (Menstrual / Follicular / Fertility / Luteal).

IMPORTANT: hormone metabolite columns (lh/estrogen/pdg) are NOT used as features when
predicting phase — they are part of how Mira derives phase and would leak the label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MCPHASES = ROOT / "data" / "mcphases"

ORDINAL = {
    "Very Low/Little": 0, "Low": 1, "Below Average": 2, "Average": 3,
    "Above Average": 4, "High": 5, "Very High": 6,
}
SYMPTOM_COLS = [
    "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
    "moodswing", "stress", "foodcravings", "indigestion", "bloating", "appetite",
]
PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


def _ord(s: pd.Series) -> pd.Series:
    return s.map(ORDINAL)


def _day_agg(path: Path, value_col: str, out_name: str, how: str = "mean") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "study_interval", "day_in_study", out_name])
    df = pd.read_csv(path, usecols=lambda c: c in {
        "id", "study_interval", "day_in_study", value_col, "timestamp"
    } or c in ("id", "study_interval", "day_in_study", value_col))
    if "day_in_study" not in df.columns:
        return pd.DataFrame(columns=["id", "study_interval", "day_in_study", out_name])
    g = df.groupby(["id", "study_interval", "day_in_study"], as_index=False)[value_col]
    if how == "sum":
        out = g.sum()
    elif how == "std":
        out = g.std()
    else:
        out = g.mean()
    return out.rename(columns={value_col: out_name})


def build_mcphases_table(mcphases_dir: Path | None = None) -> pd.DataFrame:
    d = Path(mcphases_dir) if mcphases_dir else MCPHASES
    if not (d / "hormones_and_selfreport.csv").exists():
        raise FileNotFoundError(
            f"mcPHASES not found at {d}. Place restricted CSVs there (gitignored)."
        )

    base = pd.read_csv(d / "hormones_and_selfreport.csv")
    base = base[base["phase"].isin(PHASES)].copy()
    for c in SYMPTOM_COLS:
        if c in base.columns:
            base[c + "_ord"] = _ord(base[c])

    # Sleep: join on sleep_end day (morning of)
    sleep_p = d / "sleep.csv"
    if sleep_p.exists():
        sl = pd.read_csv(
            sleep_p,
            usecols=[
                "id", "study_interval", "sleep_end_day_in_study",
                "minutesasleep", "minutesawake", "efficiency", "mainsleep",
            ],
        )
        sl = sl[sl.get("mainsleep", True) == True]  # noqa: E712
        sl = sl.rename(columns={"sleep_end_day_in_study": "day_in_study"})
        sl_agg = sl.groupby(["id", "study_interval", "day_in_study"], as_index=False).agg(
            sleep_minutes=("minutesasleep", "sum"),
            sleep_awake=("minutesawake", "sum"),
            sleep_efficiency=("efficiency", "mean"),
        )
        base = base.merge(sl_agg, on=["id", "study_interval", "day_in_study"], how="left")

    rhr = _day_agg(d / "resting_heart_rate.csv", "value", "resting_hr")
    base = base.merge(rhr, on=["id", "study_interval", "day_in_study"], how="left")

    steps = _day_agg(d / "steps.csv", "steps", "steps_sum", how="sum")
    base = base.merge(steps, on=["id", "study_interval", "day_in_study"], how="left")

    stress = _day_agg(d / "stress_score.csv", "stress_score", "stress_score_mean")
    base = base.merge(stress, on=["id", "study_interval", "day_in_study"], how="left")

    temp = _day_agg(
        d / "wrist_temperature.csv", "temperature_diff_from_baseline", "wrist_temp_delta"
    )
    base = base.merge(temp, on=["id", "study_interval", "day_in_study"], how="left")

    gluc = _day_agg(d / "glucose.csv", "glucose_value", "glucose_mean")
    base = base.merge(gluc, on=["id", "study_interval", "day_in_study"], how="left")

    hrv_p = d / "heart_rate_variability_details.csv"
    if hrv_p.exists():
        hrv = pd.read_csv(hrv_p, usecols=["id", "study_interval", "day_in_study", "rmssd"])
        hrv = hrv.groupby(["id", "study_interval", "day_in_study"], as_index=False)["rmssd"].mean()
        hrv = hrv.rename(columns={"rmssd": "hrv_rmssd"})
        base = base.merge(hrv, on=["id", "study_interval", "day_in_study"], how="left")

    feature_cols = (
        [c + "_ord" for c in SYMPTOM_COLS if c + "_ord" in base.columns]
        + [
            c for c in [
                "sleep_minutes", "sleep_awake", "sleep_efficiency",
                "resting_hr", "steps_sum", "stress_score_mean",
                "wrist_temp_delta", "glucose_mean", "hrv_rmssd", "is_weekend",
            ]
            if c in base.columns
        ]
    )
    # ensure numeric
    if "is_weekend" in feature_cols:
        base["is_weekend"] = base["is_weekend"].astype(float)

    out = base[["id", "study_interval", "day_in_study", "phase"] + feature_cols].copy()
    out.attrs["feature_cols"] = feature_cols
    return out


def matrix_from_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_cols = df.attrs.get("feature_cols") or [
        c for c in df.columns if c not in ("id", "study_interval", "day_in_study", "phase")
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["phase"].to_numpy()
    pid = df["id"].to_numpy()
    return X, y, pid, feature_cols
