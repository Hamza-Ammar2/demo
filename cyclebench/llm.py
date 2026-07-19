"""Optional LLM layer — the ONLY place a language model is used.

Design invariant (see docs/MEDICAL_SAFETY.md, ARCHITECTURE.md):
  * The LLM never computes a finding, probability, or medical conclusion.
  * It is used for:
      1. extract_events_from_text: free text -> STRUCTURED intake hints (validated,
         never trusted as fact; the deterministic engine still does the analysis)
      2. rephrase_opening: rewrite the engine's opening statement in warmer plain
         language WITHOUT adding claims.
      3. compose_doctor_followup: turn *already-computed* Model-pFL phase output +
         user intake into warm doctor-facing follow-up language (questions /
         how-to-explain). The LLM must not invent a new phase or diagnosis.
  * Offline-first: with no OPENAI_API_KEY, everything degrades gracefully to
    deterministic behavior so the demo always works.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from cyclebench.intake import SYMPTOM_LABELS
from cyclebench.safety import find_violations

_MODEL = os.environ.get("CYCLEBENCH_LLM_MODEL", "gpt-4o-mini")


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


_EXTRACT_SYSTEM = (
    "You convert a person's free-text description of how they've been feeling into "
    "STRUCTURED JSON for a women's-health timeline tool. You do NOT diagnose, explain "
    "causes, or give advice. Only extract what is stated. Output strict JSON with keys: "
    "symptoms (list of {type, severity, days_ago}), last_period_days_ago (int|null), "
    "contraception_status ('none'|'on_stable'|'changed'|null), sleep_quality "
    "('ok'|'rough'|'bad'|null). Allowed symptom types: "
    + ", ".join(sorted(SYMPTOM_LABELS.keys()))
    + ". severity is one of mild|moderate|severe. If a value is not stated, use null. "
    "Never invent dates or symptoms."
)


def extract_events_from_text(text: str) -> dict:
    """Free text -> structured intake hints. Falls back to a keyword heuristic offline."""
    text = (text or "").strip()
    if not text:
        return {}
    client = _client()
    if client is None:
        return _keyword_fallback(text)
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return _sanitize(data)
    except Exception:
        return _keyword_fallback(text)


def _sanitize(data: dict) -> dict:
    out: dict = {}
    syms = []
    for s in (data.get("symptoms") or [])[:12]:
        t = s.get("type")
        if t in SYMPTOM_LABELS:
            sev = s.get("severity")
            sev = sev if sev in ("mild", "moderate", "severe") else "moderate"
            da = s.get("days_ago")
            syms.append({"type": t, "severity": sev,
                         "days_ago": int(da) if isinstance(da, (int, float)) else None})
    if syms:
        out["symptoms"] = syms
    if isinstance(data.get("last_period_days_ago"), (int, float)):
        out["last_period_days_ago"] = int(data["last_period_days_ago"])
    if data.get("contraception_status") in ("none", "on_stable", "changed"):
        out["contraception_status"] = data["contraception_status"]
    if data.get("sleep_quality") in ("ok", "rough", "bad"):
        out["sleep_quality"] = data["sleep_quality"]
    return out


def _keyword_fallback(text: str) -> dict:
    """Very light offline extraction so the feature works without an API key."""
    t = text.lower()
    syms = []
    kw = {
        "migraine": "migraine", "headache": "headache", "cramp": "cramps",
        "tired": "fatigue", "fatigue": "fatigue", "exhaust": "fatigue",
        "brain fog": "brain_fog", "foggy": "brain_fog", "mood": "mood",
        "bloat": "bloating", "breast": "sore_breasts", "nausea": "nausea",
        "pelvic": "pelvic_pain",
    }
    seen = set()
    for k, v in kw.items():
        if k in t and v not in seen:
            sev = "severe" if any(w in t for w in ("severe", "terrible", "awful", "worst")) else "moderate"
            syms.append({"type": v, "severity": sev, "days_ago": None})
            seen.add(v)
    out: dict = {}
    if syms:
        out["symptoms"] = syms
    if any(w in t for w in ("switched pill", "new pill", "changed contraception", "started the pill", "stopped the pill")):
        out["contraception_status"] = "changed"
    if any(w in t for w in ("bad sleep", "can't sleep", "cant sleep", "insomnia", "not sleeping")):
        out["sleep_quality"] = "bad"
    return out


_REPHRASE_SYSTEM = (
    "You rewrite a clinical-style opening sentence into warm, plain, first-person language "
    "for a patient to read to their doctor. STRICT RULES: do not add any new facts, numbers, "
    "diagnoses, causes, or advice. Keep it to 2-3 sentences. Keep it association-only. "
    "Never say a condition is present or caused by anything. Preserve all hedging."
)


def rephrase_opening(opening: str) -> tuple[str, bool]:
    """Return (text, used_llm). Falls back to the original if unavailable or unsafe."""
    client = _client()
    if client is None:
        return opening, False
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _REPHRASE_SYSTEM},
                {"role": "user", "content": opening},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or find_violations(text):
            return opening, False  # safety fallback
        return text, True
    except Exception:
        return opening, False


_FOLLOWUP_SYSTEM = (
    "You help a patient prepare for a doctor's visit. You receive (a) what the person "
    "said, (b) a RESEARCH cycle-phase *estimate* already computed by a model, and "
    "(c) suggested ask-doctor questions. STRICT RULES: "
    "Do NOT diagnose, treat, or claim the phase estimate is certain or causal. "
    "Do NOT invent a different phase than the one provided. "
    "Do NOT invent lab results or new symptoms. "
    "Write 3-5 short sentences in warm first person the patient could say or bring. "
    "Include how the phase estimate might contextualize their symptoms as an association "
    "only, plus 1-2 concrete follow-up questions. Preserve hedging."
)


def _deterministic_followup(
    free_text: str | None,
    intake: dict,
    phase_pred: dict,
    doctor_questions: list[str],
) -> str:
    phase = phase_pred.get("predicted_state") or "unknown"
    conf = phase_pred.get("confidence")
    conf_s = f" (about {int(round(float(conf) * 100))}%)" if conf is not None else ""
    syms = [
        (s.get("type") or "").replace("_", " ")
        for s in (intake.get("symptoms") or [])
        if s.get("type")
    ]
    sym_bit = ", ".join(syms[:4]) if syms else "the symptoms I've been noticing"
    said = (free_text or "").strip()
    lead = f"I've been dealing with {sym_bit}."
    if said:
        lead = f"Here's what I described: {said[:280]}{'…' if len(said) > 280 else ''}"
    pad = phase_pred.get("sequence_padded")
    pad_note = (
        " This phase estimate used a short history bootstrap, so treat it lightly."
        if pad else ""
    )
    q = doctor_questions[0] if doctor_questions else (
        phase_pred.get("ask_doctor")
        or "How should we interpret this cycle-phase estimate alongside my history?"
    )
    return (
        f"{lead} A research model estimates I may be in the {phase} window{conf_s} — "
        f"that's an association with symptom/wearable-style patterns, not a diagnosis."
        f"{pad_note} I'd like to ask: {q}"
    )


def compose_doctor_followup(
    *,
    free_text: str | None,
    intake: dict,
    phase_pred: dict | None,
    doctor_questions: list[str] | None = None,
) -> tuple[str | None, bool]:
    """Model-agent: phrase doctor follow-ups from Model-pFL + user input.

    Returns (text, used_llm). Never invents a phase — requires phase_pred from pFL/sklearn.
    """
    if not phase_pred or not phase_pred.get("predicted_state"):
        return None, False
    questions = list(doctor_questions or [])
    ask = phase_pred.get("ask_doctor")
    if ask and ask not in questions:
        questions = [ask] + questions
    fallback = _deterministic_followup(free_text, intake, phase_pred, questions)

    client = _client()
    if client is None:
        if find_violations(fallback):
            return None, False
        return fallback, False

    payload = {
        "user_free_text": (free_text or "")[:800],
        "intake_symptoms": intake.get("symptoms") or [],
        "phase_estimate": phase_pred.get("predicted_state"),
        "phase_confidence": phase_pred.get("confidence"),
        "phase_model": phase_pred.get("model"),
        "sequence_padded": phase_pred.get("sequence_padded"),
        "suggested_questions": questions[:4],
        "model_statement": phase_pred.get("statement"),
    }
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or find_violations(text):
            return fallback, False
        # Guard: LLM must not rename the phase
        phase = str(phase_pred.get("predicted_state") or "")
        if phase and phase.lower() not in text.lower():
            # soft: append explicit phase if omitted
            text = f"{text} (Phase estimate provided by the research model: {phase}.)"
            if find_violations(text):
                return fallback, False
        return text, True
    except Exception:
        return fallback, False
