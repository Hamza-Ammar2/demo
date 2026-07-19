"""Case Compiler API (FastAPI).

Ephemeral, in-memory storage only — no real health data is persisted. Serves the
static frontend at '/' and the deterministic CycleBench engine over HTTP.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cyclebench.audit import audit_all
from cyclebench.case import Case
from cyclebench.engine import compile_case
from cyclebench.fixtures import build_sarah_case
from cyclebench.schema import AnalysisMode

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RESULTS = ROOT / "results"

app = FastAPI(title="Case Compiler API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# In-memory, non-persistent store (cleared on restart).
_STORE: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _serialize(result, mode: AnalysisMode) -> dict:
    timeline = [
        {
            "event": e.event.model_dump(mode="json"),
            "sort_date": e.sort_date.isoformat() if e.sort_date else None,
            "order_confidence": e.order_confidence,
            "notes": e.notes,
        }
        for e in result.timeline
    ]
    return {
        "mode": mode.value,
        "brief": result.brief.model_dump(mode="json"),
        "findings": [f.model_dump(mode="json") for f in result.findings],
        "timeline": timeline,
    }


class CompileRequest(BaseModel):
    case: Case
    mode: AnalysisMode = AnalysisMode.retrospective


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "case-compiler", "version": "0.1.0"}


@app.post("/cases/compile")
def compile_endpoint(req: CompileRequest) -> dict:
    result = compile_case(req.case, req.mode)
    payload = _serialize(result, req.mode)
    case_id = str(uuid.uuid4())[:8]
    _STORE[case_id] = payload
    payload["case_id"] = case_id
    return payload


@app.get("/demo/sarah")
def demo_sarah(mode: str = "retrospective") -> dict:
    try:
        m = AnalysisMode(mode)
    except ValueError:
        raise HTTPException(400, f"invalid mode '{mode}'")
    result = compile_case(build_sarah_case(), m)
    payload = _serialize(result, m)
    case_id = "sarah-" + m.value[:5]
    _STORE[case_id] = payload
    payload["case_id"] = case_id
    return payload


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    if case_id not in _STORE:
        raise HTTPException(404, "case not found (in-memory store; may have been cleared)")
    return _STORE[case_id]


@app.get("/cases/{case_id}/doctor-brief")
def get_brief(case_id: str) -> dict:
    if case_id not in _STORE:
        raise HTTPException(404, "case not found")
    return _STORE[case_id]["brief"]


@app.get("/cases/{case_id}/timeline")
def get_timeline(case_id: str) -> dict:
    if case_id not in _STORE:
        raise HTTPException(404, "case not found")
    return {"timeline": _STORE[case_id]["timeline"]}


@app.get("/benchmark/results")
def benchmark_results() -> dict:
    import json
    p = RESULTS / "benchmark_results.json"
    if not p.exists():
        from cyclebench.benchmark.runner import evaluate
        return evaluate()
    return json.loads(p.read_text())


@app.post("/audit/run")
def audit_run() -> dict:
    from cyclebench.audit import _honest_split, _leaking_split
    findings = compile_case(build_sarah_case(), AnalysisMode.retrospective).findings
    honest = audit_all(findings, _honest_split())
    leaking = audit_all(findings, _leaking_split())
    return {
        "honest_split": {"passed": honest.passed, "checks": honest.checks},
        "leaking_split": {"passed": leaking.passed, "checks": leaking.checks},
        "audit_behaves_correctly": honest.passed and not leaking.passed,
    }


@app.get("/mcphases/validation")
def mcphases_validation() -> dict:
    import json
    p = RESULTS / "mcphases_validation.json"
    if not p.exists():
        raise HTTPException(404, "run `make mcphases` first (requires local restricted data)")
    return json.loads(p.read_text())


class HormonalStateRequest(BaseModel):
    features: dict[str, float]


class MenopauseStageRequest(BaseModel):
    features: dict[str, float]


@app.post("/models/hormonal-state")
def model_hormonal_state(req: HormonalStateRequest) -> dict:
    from cyclebench.model.predict import hormonal_state_to_finding, predict_hormonal_state
    try:
        pred = predict_hormonal_state(req.features)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    finding = hormonal_state_to_finding(pred)
    return {"prediction": pred, "finding": finding.model_dump(mode="json")}


@app.post("/models/menopause-stage")
def model_menopause_stage(req: MenopauseStageRequest) -> dict:
    from cyclebench.model.predict import menopause_stage_to_finding, predict_menopause_stage
    try:
        pred = predict_menopause_stage(req.features)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    finding = menopause_stage_to_finding(pred)
    return {"prediction": pred, "finding": finding.model_dump(mode="json")}


@app.get("/models/metrics")
def model_metrics() -> dict:
    import json
    out = {}
    for name in ("model_hormonal_state.json", "model_menopause_stage.json", "model_train_summary.json"):
        p = RESULTS / name
        if p.exists():
            out[name] = json.loads(p.read_text())
    if not out:
        raise HTTPException(404, "run `make train-models` first")
    return out


@app.post("/models/train-local")
def model_train_local() -> dict:
    from cyclebench.model.pfl import train_local_pfl
    res = train_local_pfl()
    if not res.get("ok"):
        raise HTTPException(400, res.get("error"))
    return res


@app.post("/models/federated-sync")
def model_federated_sync() -> dict:
    from cyclebench.model.pfl import federated_sync_pfl
    res = federated_sync_pfl()
    if not res.get("ok"):
        raise HTTPException(400, res.get("error"))
    return res


class ConsentRequest(BaseModel):
    consent: bool


@app.post("/models/consent")
def model_consent(req: ConsentRequest) -> dict:
    from cyclebench.model.pfl import set_user_consent
    set_user_consent(req.consent)
    return {"ok": True, "consent": req.consent}


@app.post("/models/huggingface-sync")
def model_huggingface_sync() -> dict:
    from cyclebench.model.pfl import sync_local_to_huggingface
    res = sync_local_to_huggingface()
    if not res.get("ok"):
        raise HTTPException(403, res.get("error"))
    return res


# Static frontend (served last so API routes take precedence).
if WEB.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB / "index.html"))

    @app.get("/input")
    def page_input() -> FileResponse:
        return FileResponse(str(WEB / "input.html"))

    @app.get("/analyse")
    def page_analyse() -> FileResponse:
        return FileResponse(str(WEB / "analyse.html"))

    @app.get("/analyse-full")
    def page_analyse_full() -> FileResponse:
        return FileResponse(str(WEB / "analyse-full.html"))

    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
