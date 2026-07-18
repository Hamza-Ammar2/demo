"""Confounder detector.

Identifies competing explanations that should weaken a temporal finding:
poor sleep, medication/contraception changes, stress, and data-quality issues.
"""

from __future__ import annotations

from datetime import timedelta

from cyclebench.schema import EventType, HealthEvent

# Sleep below this (hours) near an episode is treated as a plausible confounder.
LOW_SLEEP_HOURS = 6.0
# How close (days) a confounding measurement must be to an episode to "overlap".
OVERLAP_DAYS = 1


def _overlaps(a_date, b_date, days: int = OVERLAP_DAYS) -> bool:
    return abs((a_date - b_date).days) <= days


def detect_confounders(episodes: list[HealthEvent], events: list[HealthEvent]) -> list[dict]:
    """Return a list of confounder descriptors, each with the events it implicates."""
    confounders: list[dict] = []
    dated_episodes = [e for e in episodes if e.start is not None]

    # 1. Poor sleep overlapping episodes.
    sleep_events = [
        e for e in events
        if e.event_type == EventType.sleep_measurement and e.value is not None
        and e.start is not None
    ]
    low_sleep_hits = []
    for ep in dated_episodes:
        for s in sleep_events:
            if s.value < LOW_SLEEP_HOURS and _overlaps(ep.start, s.start):
                low_sleep_hits.append((ep.event_id, s.event_id))
                break
    if low_sleep_hits:
        confounders.append({
            "type": "poor_sleep",
            "n_overlapping_episodes": len(low_sleep_hits),
            "n_episodes": len(dated_episodes),
            "episode_event_ids": [ep for ep, _ in low_sleep_hits],
            "confounder_event_ids": [s for _, s in low_sleep_hits],
            "statement": (
                f"Reduced sleep (<{LOW_SLEEP_HOURS:g}h) overlapped "
                f"{len(low_sleep_hits)} of {len(dated_episodes)} episodes."
            ),
        })

    # 2. Medication / contraception / dose changes anywhere in the record.
    change_types = {
        EventType.medication_started,
        EventType.medication_stopped,
        EventType.dose_changed,
        EventType.contraception_changed,
    }
    change_events = [e for e in events if e.event_type in change_types]
    if change_events:
        confounders.append({
            "type": "medication_or_contraception_change",
            "n_changes": len(change_events),
            "confounder_event_ids": [e.event_id for e in change_events],
            "statement": (
                f"{len(change_events)} medication/contraception change(s) occurred during "
                "the observed period and could contribute to symptom variation."
            ),
        })

    # 3. Elevated stress overlapping episodes.
    stress_events = [
        e for e in events
        if e.event_type == EventType.stress_entry and e.severity is not None
        and e.start is not None
    ]
    stress_hits = [
        (ep.event_id, s.event_id)
        for ep in dated_episodes for s in stress_events
        if s.severity >= 7 and _overlaps(ep.start, s.start)
    ]
    if stress_hits:
        confounders.append({
            "type": "high_stress",
            "n_overlapping_episodes": len(stress_hits),
            "confounder_event_ids": [s for _, s in stress_hits],
            "statement": (
                f"Elevated stress overlapped {len(stress_hits)} episode(s)."
            ),
        })

    return confounders
