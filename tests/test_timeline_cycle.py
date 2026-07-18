from datetime import date

from cyclebench.engine.cycle import (
    assign_cycle_position,
    cycle_regularity,
    estimate_cycle_lengths,
)
from cyclebench.engine.timeline import compile_timeline
from cyclebench.schema import AnalysisMode, DatePrecision, EventType, HealthEvent


def _ev(eid, d, precision=DatePrecision.day, etype=EventType.symptom):
    return HealthEvent(event_id=eid, subject_id="s", event_type=etype,
                       start=d, date_precision=precision)


def test_timeline_orders_chronologically_and_puts_undated_last():
    events = [
        _ev("b", date(2024, 3, 1)),
        _ev("a", date(2024, 1, 1)),
        HealthEvent(event_id="z", subject_id="s", event_type=EventType.free_text_note),
        _ev("c", date(2024, 2, 1)),
    ]
    tl = compile_timeline(events)
    ids = [e.event.event_id for e in tl]
    assert ids[:3] == ["a", "c", "b"]
    assert ids[-1] == "z"
    assert tl[-1].order_confidence == "undated"


def test_timeline_flags_approximate_dates():
    tl = compile_timeline([_ev("x", date(2024, 1, 1), DatePrecision.approximate)])
    assert tl[0].order_confidence == "approximate"
    assert any("approximate" in n for n in tl[0].notes)


def test_estimate_cycle_lengths():
    onsets = [date(2024, 1, 1), date(2024, 1, 29), date(2024, 2, 26)]
    assert estimate_cycle_lengths(onsets) == [28, 28]


def test_regularity_none_when_insufficient():
    assert cycle_regularity([date(2024, 1, 1)]) is None


def test_retrospective_uses_future_causal_does_not():
    onsets = [date(2024, 1, 1), date(2024, 1, 29)]
    target = date(2024, 1, 20)
    retro = assign_cycle_position(target, onsets, AnalysisMode.retrospective)
    causal = assign_cycle_position(target, onsets, AnalysisMode.causal)
    assert retro.used_future_data is True
    assert causal.used_future_data is False
    # causal cannot know cycle length from a single prior onset
    assert causal.cycle_length is None


def test_causal_estimates_from_prior_cycles():
    onsets = [date(2024, 1, 1), date(2024, 1, 29), date(2024, 2, 26)]
    pos = assign_cycle_position(date(2024, 3, 10), onsets, AnalysisMode.causal)
    assert pos.cycle_length == 28
    assert pos.phase is not None
    assert pos.used_future_data is False


def test_mode_is_mandatory():
    import pytest
    with pytest.raises(ValueError):
        assign_cycle_position(date(2024, 1, 10), [date(2024, 1, 1)], None)
