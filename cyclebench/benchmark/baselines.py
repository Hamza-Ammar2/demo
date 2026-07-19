"""Baselines for CycleBench-Bench.

Path A — Naive generic summarizer (stand-in for an unconstrained chatbot summary):
  * Claims a hormonal cyclical pattern whenever ANY symptom label repeats >= 3 times,
    with no cycle alignment, no confounder check, and no provenance.
  * Phrases the claim causally ("...are caused by hormonal changes"), i.e. it makes an
    unsupported causal/medical claim.
  This baseline is intentionally simple and is documented as illustrative in
  docs/BENCHMARK.md — it represents the failure mode the challenge calls a weak submission.

Path B — CycleBench engine (deterministic, provenance-linked, safety-guarded).
"""

from __future__ import annotations

from collections import Counter

from cyclebench.case import Case
from cyclebench.engine import compile_case
from cyclebench.schema import AnalysisMode, EstablishmentClass, EventType, FindingType


def naive_summary(case: Case) -> dict:
    """Return the naive summarizer's 'analysis' of a case (path A)."""
    labels = Counter(
        (e.label or e.event_type.value)
        for e in case.events if e.event_type == EventType.symptom
    )
    repeats = any(c >= 3 for c in labels.values())
    claims_pattern = repeats
    statements = []
    if claims_pattern:
        top = labels.most_common(1)[0][0]
        # Unsupported, causal, no provenance.
        statements.append(f"Your recurring {top} is caused by hormonal cycle changes.")
    return {
        "claims_pattern": claims_pattern,
        "statements": statements,
        "detected_confounders": set(),      # naive path detects none
        "detected_missing": set(),          # naive path detects none
        "detected_change_after_event": False,
        "provenance_present": False,        # naive path attaches no provenance
    }


def engine_analysis(case: Case, mode: AnalysisMode = AnalysisMode.retrospective) -> dict:
    """Return the CycleBench engine's analysis of a case (path B)."""
    result = compile_case(case, mode)
    findings = result.findings

    cyc = next((f for f in findings if f.finding_id == "F_cyclical"), None)
    claims_pattern = False
    if cyc is not None:
        m = cyc.metrics
        claims_pattern = (
            (m.get("relative_frequency") or 0) >= 1.5
            and m.get("n_in_window", 0) >= 3
            and m.get("confidence") in ("medium", "high")
        )

    detected_confounders = set()
    for f in findings:
        if f.finding_type == FindingType.confounder and "sleep" in f.title.lower():
            detected_confounders.add("poor_sleep")

    detected_missing = {
        f.metrics.get("field")
        for f in findings
        if f.establishment == EstablishmentClass.missing and f.metrics.get("field")
    }

    change = next((f for f in findings if f.finding_id == "F_change_after_contraception"), None)
    detected_change = bool(change and change.metrics.get("direction") == "increase")

    # provenance present iff every asserting finding carries provenance
    asserting = [f for f in findings
                 if f.establishment in (EstablishmentClass.established, EstablishmentClass.possible)]
    provenance_present = all(f.supporting_event_ids or f.source_ids for f in asserting)

    return {
        "claims_pattern": claims_pattern,
        "statements": [f.statement for f in asserting],
        "detected_confounders": detected_confounders,
        "detected_missing": detected_missing,
        "detected_change_after_event": detected_change,
        "provenance_present": provenance_present,
        "brief": result.brief,
        "findings": findings,
    }
