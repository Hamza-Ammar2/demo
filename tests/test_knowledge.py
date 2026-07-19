"""Tests for curated medical fundamentals (now foundation-backed)."""

from cyclebench.knowledge import match_fundamentals
from cyclebench.safety import assert_safe


def test_estrogen_headache_fundamental_matches():
    hits = match_fundamentals({
        "symptoms": [{"type": "migraine", "severity": "severe"}],
        "contraception_status": "on_stable",
        "contraception_formulation": "Combined pill",
        "symptom_timing": "before_period",
    })
    ids = {h["id"] for h in hits}
    assert "assoc.estrogen_migraine" in ids
    for h in hits:
        assert_safe(h["talking_point"])
        assert_safe(h["ask_doctor"])
        assert h["source"]


def test_no_fundamental_when_unrelated():
    hits = match_fundamentals({
        "symptoms": [{"type": "bloating", "severity": "mild"}],
        "contraception_status": "none",
        "sleep_quality": "ok",
        "age_range": "20-29",
    })
    # bloating alone may still match luteal bloating edge — that's OK if timing absent
    # with tightened matcher, bloating without timing should not force contraception edges
    ids = {h["id"] for h in hits}
    assert "assoc.estrogen_migraine" not in ids
    assert "assoc.amenorrhea_eval" not in ids


def test_amenorrhea_and_sleep_overlap():
    hits = match_fundamentals({
        "symptoms": [{"type": "headache", "severity": "moderate"}],
        "last_period_days_ago": 120,
        "sleep_quality": "bad",
    })
    ids = {h["id"] for h in hits}
    assert "assoc.amenorrhea_eval" in ids
    assert "assoc.sleep_headache" in ids
