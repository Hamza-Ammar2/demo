"""Tests for personalized sequence research + Research depth panel API."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyclebench.model.sequence_research import (
    DISPLAY_NAME,
    TASK_ID,
    build_soft_read_signal,
)
from cyclebench.safety import assert_safe

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "pfl_multi_symptom.json"


def test_display_name_is_not_jargon():
    assert "pFL" not in DISPLAY_NAME
    assert "sequence" in DISPLAY_NAME.lower()
    assert TASK_ID == "sequence_research"


def test_cohort_benchmark_signal_when_metrics_exist():
    if not METRICS.exists():
        pytest.skip("run make pfl-smoke first")
    sig = build_soft_read_signal({"symptoms": [{"type": "cramps", "severity": "moderate"}]})
    assert sig is not None
    assert sig["task"] == TASK_ID
    assert sig["mode"] == "cohort_benchmark"
    assert_safe(sig["statement"])


def test_no_signal_without_metrics_or_window(tmp_path, monkeypatch):
    import cyclebench.model.sequence_research as sr
    monkeypatch.setattr(sr, "METRICS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(sr, "CHECKPOINT", tmp_path / "missing.pt")
    assert build_soft_read_signal({}) is None


def test_feeling_off_keeps_research_out_of_soft_read():
    """Research depth is opt-in — must not appear in default model_signals."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/analyse/feeling-off", json={
        "intake": {
            "symptoms": [{"type": "cramps", "severity": "moderate"}],
            "last_period_days_ago": 14,
            "age_range": "30-39",
        },
        "use_llm": False,
        "consent": False,
    })
    assert r.status_code == 200
    body = r.json()
    signals = (body.get("foundation") or {}).get("model_signals") or []
    assert not any(s.get("task") == "sequence_research" for s in signals)
    assert "sequence_research" not in body or body.get("sequence_research") is None
    assert "research_depth_available" in body


def test_research_depth_endpoint():
    if not METRICS.exists():
        pytest.skip("run make pfl-smoke first")
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/research/depth")
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == DISPLAY_NAME
    assert len(d["comparison"]) == 3
    assert "not a diagnosis" in d["lede"].lower()
