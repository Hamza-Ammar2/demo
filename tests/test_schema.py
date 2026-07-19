from datetime import date

import pytest
from pydantic import ValidationError

from cyclebench.schema import (
    AnalysisMode,
    CycleContext,
    DatePrecision,
    EstablishmentClass,
    EventType,
    Finding,
    FindingType,
    HealthEvent,
    export_json_schema,
)


def test_event_range_validation():
    with pytest.raises(ValidationError):
        HealthEvent(
            event_id="e", subject_id="s", event_type=EventType.symptom,
            start=date(2024, 2, 1), end=date(2024, 1, 1),
            date_precision=DatePrecision.range,
        )


def test_range_precision_requires_both_ends():
    with pytest.raises(ValidationError):
        HealthEvent(
            event_id="e", subject_id="s", event_type=EventType.symptom,
            start=date(2024, 1, 1), date_precision=DatePrecision.range,
        )


def test_causal_cycle_forbids_future_data():
    # Referencing next_known_onset in causal mode must fail loudly.
    with pytest.raises(ValidationError):
        CycleContext(
            cycle_id="c", subject_id="s",
            period_onset=date(2024, 1, 1), next_known_onset=date(2024, 1, 29),
            analysis_mode=AnalysisMode.causal,
        )
    with pytest.raises(ValidationError):
        CycleContext(
            cycle_id="c", subject_id="s", period_onset=date(2024, 1, 1),
            analysis_mode=AnalysisMode.causal, used_future_data=True,
        )


def test_retrospective_cycle_allows_future_data():
    ctx = CycleContext(
        cycle_id="c", subject_id="s",
        period_onset=date(2024, 1, 1), next_known_onset=date(2024, 1, 29),
        analysis_mode=AnalysisMode.retrospective, used_future_data=True,
    )
    assert ctx.used_future_data is True


def test_asserting_finding_requires_provenance():
    with pytest.raises(ValidationError):
        Finding(
            finding_id="f", title="t", statement="something",
            finding_type=FindingType.temporal_association,
            method="m", analysis_mode=AnalysisMode.retrospective,
            establishment=EstablishmentClass.possible,
        )


def test_missing_finding_needs_no_provenance():
    f = Finding(
        finding_id="f", title="t", statement="a field is missing",
        finding_type=FindingType.missing_information,
        method="m", analysis_mode=AnalysisMode.retrospective,
        establishment=EstablishmentClass.missing,
    )
    assert f.establishment == EstablishmentClass.missing


def test_json_schema_export_has_all_entities():
    schema = export_json_schema()
    for name in ["SubjectProfile", "HealthEvent", "SourceReference",
                 "CycleContext", "Finding", "DoctorBrief"]:
        assert name in schema
        assert schema[name]["type"] == "object"
