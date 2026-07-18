import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_demo_sarah_retrospective_and_store():
    r = client.get("/demo/sarah?mode=retrospective")
    assert r.status_code == 200
    d = r.json()
    cyc = next(f for f in d["findings"] if f["finding_id"] == "F_cyclical")
    assert cyc["metrics"]["n_in_window"] == 4
    assert d["brief"]["analysis_mode"] == "retrospective"
    # stored and retrievable
    cid = d["case_id"]
    assert client.get(f"/cases/{cid}/doctor-brief").status_code == 200
    assert client.get(f"/cases/{cid}/timeline").json()["timeline"]


def test_demo_sarah_causal_is_conservative():
    d = client.get("/demo/sarah?mode=causal").json()
    cyc = next(f for f in d["findings"] if f["finding_id"] == "F_cyclical")
    assert cyc["metrics"]["n_in_window"] <= 4


def test_invalid_mode_rejected():
    assert client.get("/demo/sarah?mode=bogus").status_code == 400


def test_audit_endpoint_rejects_leak():
    d = client.post("/audit/run").json()
    assert d["audit_behaves_correctly"] is True
    assert d["honest_split"]["passed"] and not d["leaking_split"]["passed"]


def test_benchmark_endpoint():
    s = client.get("/benchmark/results").json()["summary"]
    assert s["pattern_detection_accuracy"]["path_B_cyclebench"] >= 0.9


def test_missing_case_404():
    assert client.get("/cases/nope/timeline").status_code == 404
