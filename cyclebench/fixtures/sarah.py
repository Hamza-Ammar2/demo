"""The golden 'Sarah' case — pre-extracted structured events (NO LLM required).

Sarah, age 34: recurrent migraines, fatigue, brain fog across five cycles.
Designed so the deterministic engine discovers:
  - 4 of 5 severe migraines fall in the same (luteal) cycle window,
  - reduced sleep overlaps 3 of them (a confounder),
  - symptom frequency rises after a contraception change,
  - contraceptive formulation + one period date + pre-change baseline are missing.

This fixture is SYNTHETIC and labeled as such via its SourceReference.
"""

from __future__ import annotations

from datetime import date

from cyclebench.case import Case
from cyclebench.schema import (
    AnalysisMode,
    Certainty,
    CycleContext,
    DatePrecision,
    EvidenceClass,
    EventType,
    HealthEvent,
    SourceReference,
    SourceType,
    SubjectProfile,
)

SRC = "sarah_diary"


def _sym(eid, d, label, sev):
    return HealthEvent(
        event_id=eid, subject_id="sarah", event_type=EventType.symptom,
        label=label, start=d, date_precision=DatePrecision.day, severity=sev,
        certainty=Certainty.medium, evidence_class=EvidenceClass.patient_reported,
        source_id=SRC, observed_at=None,
    )


def _onset(eid, d, precision=DatePrecision.day):
    return HealthEvent(
        event_id=eid, subject_id="sarah", event_type=EventType.menstrual_onset,
        label="period onset", start=d, date_precision=precision,
        certainty=Certainty.high if precision == DatePrecision.day else Certainty.low,
        evidence_class=EvidenceClass.patient_reported, source_id=SRC,
    )


def _sleep(eid, d, hours):
    return HealthEvent(
        event_id=eid, subject_id="sarah", event_type=EventType.sleep_measurement,
        label="sleep hours", start=d, date_precision=DatePrecision.day,
        value=hours, unit="hours", certainty=Certainty.high,
        evidence_class=EvidenceClass.measured, source_id="sarah_wearable",
    )


def build_sarah_case() -> Case:
    subject = SubjectProfile(
        subject_id="sarah",
        age_range="30-39",
        life_stage="reproductive",
        contraception_status="changed_during_period",
        contraception_formulation=None,  # deliberately missing
        source_quality=Certainty.medium,
    )

    onsets = [
        _onset("o1", date(2024, 1, 3)),
        _onset("o2", date(2024, 1, 31)),
        _onset("o3", date(2024, 2, 29)),
        _onset("o4", date(2024, 3, 28)),
        _onset("o5", date(2024, 4, 26)),
        _onset("o6", date(2024, 5, 24)),
        # incomplete cycle: an approximate/uncertain period date
        _onset("o7", date(2024, 6, 21), precision=DatePrecision.approximate),
    ]

    # Severe migraines: 4 in the luteal window (~day 22-24), 1 outside (mid-cycle).
    migraines = [
        _sym("m1", date(2024, 1, 25), "severe migraine", 8),   # cycle1 luteal
        _sym("m2", date(2024, 2, 24), "severe migraine", 9),   # cycle2 luteal
        _sym("m3", date(2024, 3, 22), "severe migraine", 8),   # cycle3 luteal
        _sym("m4", date(2024, 4, 20), "severe migraine", 9),   # cycle4 luteal
        _sym("m5", date(2024, 5, 10), "severe migraine", 8),   # cycle5 mid-cycle (the odd one)
    ]

    other_symptoms = [
        _sym("s1", date(2024, 2, 10), "fatigue", 5),
        _sym("s2", date(2024, 3, 15), "brain fog", 4),
        _sym("s3", date(2024, 4, 5), "fatigue", 5),
    ]

    # Reduced sleep (<6h) overlapping 3 of the severe migraines.
    sleep = [
        _sleep("sl1", date(2024, 1, 25), 5.0),
        _sleep("sl2", date(2024, 2, 24), 5.5),
        _sleep("sl3", date(2024, 4, 20), 4.5),
        _sleep("sl4", date(2024, 3, 1), 7.5),  # normal night, no overlap
    ]

    contraception = HealthEvent(
        event_id="c1", subject_id="sarah",
        event_type=EventType.contraception_changed,
        label="switched hormonal contraception", start=date(2023, 12, 20),
        date_precision=DatePrecision.day, certainty=Certainty.medium,
        evidence_class=EvidenceClass.patient_reported, source_id=SRC,
    )
    dose_change = HealthEvent(
        event_id="d1", subject_id="sarah", event_type=EventType.dose_changed,
        label="medication dose changed", start=date(2024, 3, 1),
        date_precision=DatePrecision.day, certainty=Certainty.low,
        evidence_class=EvidenceClass.patient_reported, source_id=SRC,
        medication_context=None,  # deliberately missing name/dose
    )

    events = onsets + migraines + other_symptoms + sleep + [contraception, dose_change]

    sources = [
        SourceReference(
            source_id=SRC, source_type=SourceType.synthetic_fixture,
            excerpt="Synthetic patient diary (Sarah golden demo). Not real patient data.",
            confidence=Certainty.medium,
        ),
        SourceReference(
            source_id="sarah_wearable", source_type=SourceType.synthetic_fixture,
            excerpt="Synthetic wearable sleep stream (Sarah golden demo).",
            confidence=Certainty.high,
        ),
    ]

    cycles = [
        CycleContext(
            cycle_id=f"cyc{i}", subject_id="sarah", period_onset=o.start,
            analysis_mode=AnalysisMode.retrospective,
            anchor_source=SRC,
        )
        for i, o in enumerate(onsets)
    ]

    return Case(subject=subject, events=events, cycles=cycles, sources=sources)
