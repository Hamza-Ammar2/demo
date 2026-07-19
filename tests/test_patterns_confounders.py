from datetime import date

from cyclebench.engine.confounders import detect_confounders
from cyclebench.engine.patterns import detect_change_after_event, detect_cyclical_pattern
from cyclebench.schema import AnalysisMode, DatePrecision, EventType, HealthEvent


def _sym(eid, d, sev=8):
    return HealthEvent(event_id=eid, subject_id="s", event_type=EventType.symptom,
                       label="severe migraine", start=d, date_precision=DatePrecision.day,
                       severity=sev)


def _sleep(eid, d, hours):
    return HealthEvent(event_id=eid, subject_id="s", event_type=EventType.sleep_measurement,
                       start=d, date_precision=DatePrecision.day, value=hours, unit="hours")


# Regular ~28-day cycles.
ONSETS = [date(2024, 1, 3), date(2024, 1, 31), date(2024, 2, 29),
          date(2024, 3, 28), date(2024, 4, 26), date(2024, 5, 24)]


def test_pattern_positive_clusters_in_one_phase():
    episodes = [
        _sym("m1", date(2024, 1, 25)),
        _sym("m2", date(2024, 2, 24)),
        _sym("m3", date(2024, 3, 22)),
        _sym("m4", date(2024, 4, 20)),
    ]
    res = detect_cyclical_pattern(episodes, ONSETS, AnalysisMode.retrospective)
    assert res["dominant_phase"] == "luteal"
    assert res["n_in_window"] == 4
    assert res["relative_frequency"] >= 1.5
    assert res["confidence"] == "high"


def test_pattern_negative_no_clustering():
    # Episodes spread across the cycle -> weak/limited signal.
    episodes = [
        _sym("m1", date(2024, 1, 6)),    # early
        _sym("m2", date(2024, 2, 12)),   # mid
        _sym("m3", date(2024, 3, 26)),   # late
    ]
    res = detect_cyclical_pattern(episodes, ONSETS, AnalysisMode.retrospective)
    # No phase should dominate strongly; relative frequency near/below baseline OR low conf.
    assert res["n_in_window"] <= 2


def test_change_after_event_detects_increase():
    episodes = [
        _sym("m1", date(2024, 2, 24)),
        _sym("m2", date(2024, 3, 22)),
        _sym("m3", date(2024, 4, 20)),
    ]
    res = detect_change_after_event(episodes, date(2024, 1, 1))
    assert res["direction"] == "increase"
    assert res["n_after"] == 3 and res["n_before"] == 0


def test_confounder_detects_low_sleep_overlap():
    episodes = [_sym("m1", date(2024, 2, 24)), _sym("m2", date(2024, 3, 22))]
    events = episodes + [_sleep("sl1", date(2024, 2, 24), 5.0)]
    confs = detect_confounders(episodes, events)
    sleep = [c for c in confs if c["type"] == "poor_sleep"]
    assert sleep and sleep[0]["n_overlapping_episodes"] == 1


def test_no_confounder_when_sleep_normal():
    episodes = [_sym("m1", date(2024, 2, 24))]
    events = episodes + [_sleep("sl1", date(2024, 2, 24), 8.0)]
    confs = detect_confounders(episodes, events)
    assert not [c for c in confs if c["type"] == "poor_sleep"]
