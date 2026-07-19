"""Optional LLM layer — the ONLY place a language model is used.

Design invariant (see docs/MEDICAL_SAFETY.md, ARCHITECTURE.md):
  * The LLM never computes a finding, probability, or medical conclusion.
  * It is used for exactly two things:
      1. extract_events_from_text: free text -> STRUCTURED intake hints (validated,
         never trusted as fact; the deterministic engine still does the analysis)
      2. rephrase_opening: rewrite the engine's opening statement in warmer plain
         language WITHOUT adding claims. Output is run through the safety guard; if it
         fails or drifts, we fall back to the deterministic text.
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
