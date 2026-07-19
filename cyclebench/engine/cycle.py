"""Cycle alignment.

Given menstrual-onset dates, place an arbitrary date within its cycle and estimate
a phase — WITHOUT assuming a universal 28-day cycle or a day-14 ovulation.

Two explicit modes (the `mode` argument is mandatory everywhere it is consumed):

  retrospective: a date may be placed inside the cycle bounded by the PREVIOUS onset
                 and the NEXT known onset, so cycle length is known exactly.
  causal:        only onsets on/before the date may be used; cycle length is estimated
                 from prior cycles, so later data cannot leak backwards.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple, Optional

from cyclebench.schema import AnalysisMode

# Phase boundaries as fractions of the individual's own cycle length.
# Deliberately coarse and documented — not a clinical claim.
_PHASE_BINS = [
    ("menstrual", 0.00, 0.15),
    ("follicular", 0.15, 0.45),
    ("periovulatory", 0.45, 0.60),
    ("luteal", 0.60, 1.01),
]


def estimate_cycle_lengths(onsets: list[date]) -> list[int]:
    """Consecutive gaps (days) between sorted onset dates."""
    s = sorted(set(onsets))
    return [(s[i + 1] - s[i]).days for i in range(len(s) - 1)]


def cycle_regularity(onsets: list[date]) -> Optional[float]:
    """Coefficient of variation of cycle lengths; higher = more irregular.

    Returns None if fewer than 2 cycle lengths are available.
    """
    lengths = estimate_cycle_lengths(onsets)
    if len(lengths) < 2:
        return None
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return None
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return (var**0.5) / mean


class CyclePosition(NamedTuple):
    cycle_day: Optional[int]          # days since the anchoring onset
    cycle_length: Optional[int]       # length used (known or estimated)
    phase_fraction: Optional[float]   # 0..1 position within the cycle
    phase: Optional[str]              # coarse phase label
    used_future_data: bool            # True if next onset informed the estimate
    confidence: str                   # "high" | "medium" | "low"
    note: str


def _phase_for_fraction(frac: float) -> str:
    for name, lo, hi in _PHASE_BINS:
        if lo <= frac < hi:
            return name
    return "luteal"


def assign_cycle_position(
    target: date,
    onsets: list[date],
    mode: AnalysisMode,
) -> CyclePosition:
    """Place `target` within its menstrual cycle under the given analysis mode.

    `mode` is mandatory. In causal mode, onsets after `target` are ignored entirely.
    """
    if mode is None:  # defensive: no implicit default
        raise ValueError("analysis mode is mandatory (retrospective|causal)")

    onsets = sorted(set(onsets))
    prior = [o for o in onsets if o <= target]
    later = [o for o in onsets if o > target]

    if not prior:
        return CyclePosition(
            None, None, None, None, False, "low",
            "no onset on/before target; cannot anchor cycle",
        )

    anchor = prior[-1]
    cycle_day = (target - anchor).days

    prior_lengths = estimate_cycle_lengths(prior)
    est_length = (
        round(sum(prior_lengths) / len(prior_lengths)) if prior_lengths else None
    )

    if mode == AnalysisMode.retrospective and later:
        # Known cycle length from the enclosing cycle.
        cycle_length = (later[0] - anchor).days
        used_future = True
        confidence = "high"
        note = "retrospective: cycle length known from next onset"
    else:
        # Causal (or retrospective with no later onset): estimate from history.
        cycle_length = est_length
        used_future = False
        if cycle_length is None:
            return CyclePosition(
                cycle_day, None, None, None, False, "low",
                "only one prior onset; cannot estimate cycle length",
            )
        # Irregularity lowers confidence.
        cv = cycle_regularity(prior)
        if cv is not None and cv > 0.15:
            confidence = "low"
            note = f"causal estimate; irregular cycles (CV={cv:.2f}) reduce confidence"
        else:
            confidence = "medium"
            note = "causal: cycle length estimated from prior cycles"

    if not cycle_length or cycle_length <= 0:
        return CyclePosition(
            cycle_day, cycle_length, None, None, used_future, "low",
            "non-positive cycle length; cannot compute phase",
        )

    frac = max(0.0, min(1.0, cycle_day / cycle_length))
    phase = _phase_for_fraction(frac)
    return CyclePosition(
        cycle_day, cycle_length, frac, phase, used_future, confidence, note
    )
