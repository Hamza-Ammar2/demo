"""Tests for the model factory (cyclebench.model.tasks)."""

import pytest

from cyclebench.model.tasks import (
    PCOS,
    TASKS,
    build_matrix,
    load_frame,
    predict_task,
    task_to_finding,
    train_task,
)
from cyclebench.safety import assert_safe

PCOS_PRESENT = PCOS.data_path.exists()
pcos_only = pytest.mark.skipif(not PCOS_PRESENT, reason="Kaggle PCOS data not downloaded")


def test_task_registered_with_population_and_guards():
    t = TASKS["pcos_risk"]
    assert t.population and t.provenance
    assert t.does_not_support  # must declare invalid uses
    assert any("diagnosis" in s.lower() for s in t.does_not_support)


@pcos_only
def test_matrix_excludes_ids_and_target():
    df = load_frame(PCOS)
    X, y, ids, feats = build_matrix(PCOS, df)
    assert PCOS.target not in feats
    assert "Sl. No" not in feats and "Patient File No." not in feats
    assert not any(f.lower().startswith("unnamed") for f in feats)
    assert X.shape[0] == len(y)


@pcos_only
def test_pcos_trains_above_majority_baseline():
    bundle = train_task(PCOS)
    m = bundle.metrics
    # a real signal should beat guessing the majority class
    assert m["balanced_accuracy"] > 0.6
    assert m["accuracy"] >= m["majority_baseline_accuracy"]
    assert m["population"] == PCOS.population
    assert PCOS.bundle_path.exists()


@pcos_only
def test_pcos_predict_and_finding_is_safe_and_scoped():
    if not PCOS.bundle_path.exists():
        train_task(PCOS)
    pred = predict_task("pcos_risk", {
        "Follicle No. (R)": 15, "Follicle No. (L)": 14, "hair growth(Y/N)": 1,
        "Weight gain(Y/N)": 1, "AMH(ng/mL)": 8.0, "Cycle(R/I)": 4,
    })
    assert 0.0 <= pred["positive_probability"] <= 1.0
    assert pred["explanation"]
    finding = task_to_finding("pcos_risk", pred)
    assert_safe(finding.statement)                 # no diagnostic/causal language
    assert "not a diagnosis" in finding.statement.lower()
    assert PCOS.population in finding.limitation    # population declared downstream
