"""Missing-information detector.

Identifies the highest-value fields whose absence limits the case. Deterministic
rules only; we do NOT claim formal causal information gain.
"""

from __future__ import annotations

from cyclebench.case import Case
from cyclebench.schema import DatePrecision, EventType


def detect_missing_information(case: Case) -> list[dict]:
    """Return prioritized missing-information items (highest value first)."""
    items: list[dict] = []

    subj = case.subject
    has_contraception_change = bool(
        case.events_of(EventType.contraception_changed)
    )

    # 1. Contraceptive formulation (only high-value if contraception is in play).
    if has_contraception_change and not subj.contraception_formulation:
        items.append({
            "field": "contraceptive_formulation",
            "priority": "high",
            "statement": "Exact contraceptive formulation and dose are not recorded.",
        })

    # 2. Medication name/dose on dose-change events.
    for e in case.events_of(EventType.dose_changed, EventType.medication_started):
        if not e.medication_context:
            items.append({
                "field": "medication_detail",
                "priority": "high",
                "statement": f"Medication name/dose missing for event {e.event_id}.",
                "event_id": e.event_id,
            })
            break

    # 3. Missing/approximate period dates.
    onsets = case.events_of(EventType.menstrual_onset)
    approx_or_missing = [
        e for e in onsets
        if e.start is None or e.date_precision in (DatePrecision.approximate, DatePrecision.unknown)
    ]
    if approx_or_missing:
        items.append({
            "field": "menstrual_onset_date",
            "priority": "high",
            "statement": (
                f"{len(approx_or_missing)} menstrual-onset date(s) are missing or approximate, "
                "which weakens cycle alignment."
            ),
            "event_ids": [e.event_id for e in approx_or_missing],
        })

    # 4. Symptom severity absent.
    symptoms = case.events_of(EventType.symptom)
    if symptoms and all(e.severity is None for e in symptoms):
        items.append({
            "field": "symptom_severity",
            "priority": "medium",
            "statement": "Symptom severity is not recorded for any episode.",
        })

    # 5. Baseline before a treatment change.
    if has_contraception_change:
        change = case.events_of(EventType.contraception_changed)[0]
        if change.start is not None:
            baseline_symptoms = [
                e for e in symptoms if e.start is not None and e.start < change.start
            ]
            if len(baseline_symptoms) < 2:
                items.append({
                    "field": "pre_change_baseline",
                    "priority": "medium",
                    "statement": (
                        "Little symptom history exists before the contraception change, "
                        "limiting before/after comparison."
                    ),
                })

    return items
