"""Timeline compiler.

Turns unordered HealthEvents into a chronologically ordered timeline while being
honest about date precision, conflicts, and missing dates.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel

from cyclebench.schema import DatePrecision, HealthEvent


class TimelineEntry(BaseModel):
    event: HealthEvent
    sort_date: Optional[date]
    order_confidence: str  # "exact" | "approximate" | "undated"
    notes: list[str] = []


def _effective_date(event: HealthEvent) -> Optional[date]:
    """Best available date for ordering. For ranges we sort by start."""
    return event.start


def compile_timeline(events: list[HealthEvent]) -> list[TimelineEntry]:
    """Return events sorted chronologically.

    - Dated events sort by their start date.
    - Undated events are placed at the end and flagged (never silently dropped).
    - Same-date events keep a stable secondary order by event_id for determinism.
    - Conflicting duplicate events (same id, different dates) are annotated.
    """
    entries: list[TimelineEntry] = []
    seen: dict[str, date] = {}

    for ev in events:
        d = _effective_date(ev)
        notes: list[str] = []

        if ev.date_precision in (DatePrecision.approximate, DatePrecision.range):
            notes.append(f"date precision: {ev.date_precision.value}")

        if ev.event_id in seen and seen[ev.event_id] != d:
            notes.append(
                f"conflicting date for {ev.event_id}: {seen[ev.event_id]} vs {d}"
            )
        if d is not None:
            seen.setdefault(ev.event_id, d)

        if d is None:
            order_conf = "undated"
        elif ev.date_precision in (DatePrecision.exact, DatePrecision.day):
            order_conf = "exact"
        else:
            order_conf = "approximate"

        entries.append(
            TimelineEntry(event=ev, sort_date=d, order_confidence=order_conf, notes=notes)
        )

    # Dated first (chronological), undated last; stable tiebreak by event_id.
    entries.sort(
        key=lambda e: (
            e.sort_date is None,
            e.sort_date or date.max,
            e.event.event_id,
        )
    )
    return entries
