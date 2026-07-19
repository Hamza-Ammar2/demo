"""Tests for the data-grounding loop: reference stats, question plan, intake mapping."""

from pathlib import Path

from cyclebench.conversation import build_question_plan
from cyclebench.intake import (
    build_case_from_intake,
    intake_to_menopause_features,
    menopause_relevant,
)
from cyclebench.reference import load_reference, symptom_cohort_context

REF = Path(__file__).resolve().parents[1] / "results" / "reference_stats.json"


def test_reference_loads_and_is_aggregate_only():
    ref = load_reference()
    assert "symptom_phase" in ref and "menopause" in ref
    # aggregate only: no per-participant rows anywhere
    txt = REF.read_text() if REF.exists() else "{}"
    assert "participant_id" not in txt


def test_symptom_cohort_context_shape():
    ctx = symptom_cohort_context("headache")
    # only asserts structure when the reference has mcPHASES stats present
    if ctx is not None:
        assert "sentence" in ctx and "dominant_phase" in ctx
        assert isinstance(ctx["significant"], bool)


def test_unknown_symptom_has_no_cohort_context():
    assert symptom_cohort_context("nausea") is None


def test_question_plan_symptoms_first_and_annotated():
    plan = build_question_plan()
    qs = plan["questions"]
    assert qs[0]["id"] == "symptoms"
    # every question is annotated with what it maps to
    assert all("maps_to" in q for q in qs)
    # conditional questions carry their gate
    ctype = next(q for q in qs if q["id"] == "contraception_type")
    assert ctype["requires"]["contraception_status"]


def test_menopause_features_mapping():
    feats = intake_to_menopause_features({
        "age_range": "40-49", "last_period_days_ago": 95,
        "sleep_quality": "bad", "cycle_regularity": "irregular",
        "hot_flash_freq": 4, "bloodwork": "fsh_high",
    })
    assert feats["age_years"] == 45.0
    assert feats["amenorrhea_months"] > 0
    assert feats["sleep_disturbance"] == 3.0
    assert feats["fsh_miu_ml"] == 45.0


def test_menopause_relevance_gating():
    assert menopause_relevant({"age_range": "40-49"}) is True
    assert menopause_relevant({"hot_flash_freq": 4}) is True
    assert menopause_relevant({"age_range": "20-29", "symptoms": []}) is False


def test_intake_builds_valid_case():
    case = build_case_from_intake({
        "symptoms": [{"type": "headache", "severity": "severe", "days_ago": 2}],
        "last_period_days_ago": 20, "contraception_status": "changed",
        "sleep_quality": "bad",
    })
    assert case.subject.subject_id == "self"
    assert any(e.event_type.value == "menstrual_onset" for e in case.events)
