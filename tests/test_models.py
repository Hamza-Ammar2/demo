"""Tests for Layer 02 models."""

from pathlib import Path

import pytest

from cyclebench.model.features_swan import build_menopause_table, matrix_from_table
from cyclebench.model.common import MODELS_DIR, participant_split
from cyclebench.model.predict import menopause_stage_to_finding, predict_menopause_stage
from cyclebench.model.train import train_menopause_stage
from cyclebench.safety import assert_safe

MCPHASES = Path(__file__).resolve().parents[1] / "data" / "mcphases"


def test_participant_split_no_overlap():
    import numpy as np
    pid = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    tr, te = participant_split(pid, test_size=0.25, seed=0)
    assert set(pid[tr]).isdisjoint(set(pid[te]))


def test_menopause_synthetic_trains_and_predicts():
    bundle = train_menopause_stage(allow_synthetic=True)
    assert bundle.metrics["balanced_accuracy"] > 0.5
    assert (MODELS_DIR / "menopause_stage_v0.1.joblib").exists()

    pred = predict_menopause_stage({
        "age_years": 52, "fsh_miu_ml": 45, "estradiol_pg_ml": 25, "shbg_nmol_l": 48,
        "hot_flash_freq": 4, "night_sweat_freq": 3, "sleep_disturbance": 3,
        "cycle_irregularity": 2, "amenorrhea_months": 6, "bmi": 28,
    })
    assert pred["predicted_stage"] in {
        "premenopausal", "early_perimenopause", "late_perimenopause", "postmenopausal"
    }
    assert pred["explanation"]
    finding = menopause_stage_to_finding(pred)
    assert_safe(finding.statement)
    assert finding.establishment.value == "possible"


def test_menopause_table_shapes():
    df = build_menopause_table(allow_synthetic=True)
    X, y, pid, feats = matrix_from_table(df)
    assert X.shape[0] == len(y) == len(pid)
    assert len(feats) >= 5


@pytest.mark.skipif(not (MCPHASES / "hormones_and_selfreport.csv").exists(),
                    reason="restricted mcPHASES not present")
def test_hormonal_state_trains():
    from cyclebench.model.train import train_hormonal_state
    from cyclebench.model.predict import predict_hormonal_state, hormonal_state_to_finding
    bundle = train_hormonal_state()
    assert bundle.metrics["balanced_accuracy"] > 0.25  # > chance for 4-class
    pred = predict_hormonal_state({
        "headaches_ord": 5, "cramps_ord": 5, "fatigue_ord": 4, "sleepissue_ord": 4,
        "sleep_minutes": 300, "resting_hr": 80, "steps_sum": 3500,
    })
    assert pred["predicted_state"] in {"Menstrual", "Follicular", "Fertility", "Luteal"}
    assert_safe(hormonal_state_to_finding(pred).statement)
