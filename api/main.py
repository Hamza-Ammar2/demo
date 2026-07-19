"""Aestra API (FastAPI).

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

app = FastAPI(title="Aestra API", version="0.1.0")
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


class TaskPredictRequest(BaseModel):
    features: dict[str, float]


@app.get("/tasks")
def list_tasks() -> dict:
    """List model-factory tasks (population, supports/limits, metrics)."""
    from cyclebench.model.tasks import TASKS
    out = {}
    for name, t in TASKS.items():
        out[name] = {
            "version": t.version,
            "population": t.population,
            "provenance": t.provenance,
            "supports": t.supports,
            "does_not_support": t.does_not_support,
            "trained": t.bundle_path.exists(),
        }
    return out


@app.post("/tasks/{task_name}/predict")
def task_predict(task_name: str, req: TaskPredictRequest) -> dict:
    from cyclebench.model.tasks import TASKS, predict_task, task_to_finding
    if task_name not in TASKS:
        raise HTTPException(404, f"unknown task '{task_name}'")
    try:
        pred = predict_task(task_name, req.features)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    finding = task_to_finding(task_name, pred)
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


class ExtractRequest(BaseModel):
    text: str
    use_llm: bool = True


@app.post("/analyse/extract")
def analyse_extract(req: ExtractRequest) -> dict:
    """Parse ONE chat message into structured hints (LLM if available, else keyword).

    Never analyses or judges — just structures what was said so the client can build
    up a running list of noted symptoms and details.
    """
    from cyclebench.llm import extract_events_from_text, llm_available
    hints = extract_events_from_text(req.text) if req.use_llm else {}
    return {"hints": hints, "llm_available": llm_available()}


@app.get("/analyse/question-plan")
def analyse_question_plan() -> dict:
    """Data-driven question plan: questions ordered/annotated by model importances."""
    from cyclebench.conversation import build_question_plan
    return build_question_plan()


CONTRIB = ROOT / "data" / "contributions" / "sessions.jsonl"


@app.get("/foundation/stats")
def foundation_stats() -> dict:
    """Inspect the medical foundation graph composition."""
    from cyclebench.foundation.io import load_bundle
    b = load_bundle()
    return {
        "version": b.version,
        "n_entities": len(b.entities),
        "n_associations": len(b.associations),
        "n_evidence": len(b.evidence),
        "description": b.description,
    }


@app.get("/corpus/stats")
def corpus_stats() -> dict:
    """Aggregate corpus composition + how many sessions have been contributed."""
    from cyclebench.reference import load_reference
    ref = load_reference()
    n_contrib = 0
    if CONTRIB.exists():
        n_contrib = sum(1 for _ in CONTRIB.open())
    return {
        "sources": ref.get("sources", {}),
        "symptom_profiles": len(ref.get("symptom_phase", {})),
        "menopause_features": len(ref.get("menopause", {}).get("feature_importances", [])),
        "contributed_sessions": n_contrib,
    }


def _writeback_session(case, intake: dict, consent: bool) -> str | None:
    """Append a schema-valid case to the growing corpus (only with consent)."""
    if not consent:
        return None
    import json
    from datetime import datetime, timezone
    CONTRIB.parent.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:12]
    record = {
        "session_id": sid,
        "contributed_at": datetime.now(timezone.utc).isoformat(),
        "consent": True,
        "source": "feeling_off_intake",
        "case": case.model_dump(mode="json"),
    }
    with CONTRIB.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return sid


class FeelingOffRequest(BaseModel):
    intake: dict
    free_text: str | None = None
    use_llm: bool = True
    consent: bool = False


@app.post("/analyse/feeling-off")
def analyse_feeling_off(req: FeelingOffRequest) -> dict:
    """Self-report intake -> foundation-assembled grounded read.

    Steps: (1) optional LLM extracts structured hints from free text,
    (2) deterministic Case builder + engine (personal patterns),
    (3) foundation assemble_read = foundation fact + dataset evidence + personal pattern,
    (4) optional LLM phrasing (safety-guarded),
    (5) write the session back to grow the corpus (with consent).
    """
    from cyclebench.foundation.query import assemble_read
    from cyclebench.intake import build_case_from_intake
    from cyclebench.safety import assert_safe

    intake = dict(req.intake or {})

    llm_used = {"extraction": False, "rephrase": False, "available": False}
    if req.free_text and req.use_llm:
        from cyclebench.llm import extract_events_from_text, llm_available
        llm_used["available"] = llm_available()
        extracted = extract_events_from_text(req.free_text)
        for k, v in extracted.items():
            if k == "symptoms":
                intake.setdefault("symptoms", [])
                existing = {s.get("type") for s in intake["symptoms"]}
                intake["symptoms"] += [s for s in v if s.get("type") not in existing]
            else:
                intake.setdefault(k, v)
        llm_used["extraction"] = bool(extracted)

    case = build_case_from_intake(intake)
    result = compile_case(case, AnalysisMode.retrospective)
    payload = _serialize(result, AnalysisMode.retrospective)

    # (3) MEDICAL FOUNDATION — the product brain
    foundation = assemble_read(intake, personal_findings=result.findings)
    payload["foundation"] = foundation.model_dump(mode="json")

    # Backward-compatible projections for older UI blocks
    payload["fundamentals"] = [
        {
            "id": c.association_id,
            "title": c.title,
            "talking_point": c.foundation_fact,
            "ask_doctor": c.ask_doctor,
            "source": c.source,
            "when": ", ".join(c.datasets) if c.datasets else c.relation,
            "evidence_summaries": c.evidence_summaries,
            "personal_pattern": c.personal_pattern,
        }
        for c in foundation.cards
    ]
    payload["cohort_context"] = [
        {"sentence": s, "significant": True}
        for c in foundation.cards
        for s in c.evidence_summaries
        if "mcPHASES" in s or "cohort" in s.lower()
    ]
    meno = next((m for m in foundation.model_signals if m.get("task") == "menopause_stage"), None)
    payload["menopause_model"] = meno
    pcos = next((m for m in foundation.model_signals if m.get("task") == "pcos_risk"), None)
    payload["pcos_model"] = pcos

    existing = list(payload["brief"].get("unresolved_questions") or [])
    for q in foundation.doctor_questions:
        if q not in existing:
            existing.append(q)
    for m in foundation.missing_prompts:
        if m not in existing:
            existing.append(m)
    # Unique, stable order — never show the same ask-doctor line thrice.
    seen: set[str] = set()
    unique: list[str] = []
    for q in existing:
        key = q.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(q)
    payload["brief"]["unresolved_questions"] = unique[:6]

    # (4) optional LLM phrasing of the opening
    if req.use_llm:
        from cyclebench.llm import rephrase_opening
        new_open, used = rephrase_opening(result.brief.opening_statement)
        if used:
            assert_safe(new_open, where="llm_opening")
            payload["brief"]["opening_statement"] = new_open
            payload["llm_opening"] = new_open
        llm_used["rephrase"] = used

    # (5) grow the corpus (consented)
    payload["contributed_session_id"] = _writeback_session(case, intake, req.consent)

    payload["llm"] = llm_used
    payload["intake_used"] = intake
    return payload


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
