"""Pattern detector (narrow golden path).

Two transparent analyses, both returning plain dicts of numbers (no diagnosis
probability, ever):

  detect_cyclical_pattern     - do symptom episodes cluster in a cycle window?
  detect_change_after_event   - did symptom frequency change after a med/contraception event?
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from cyclebench.engine.cycle import assign_cycle_position, cycle_regularity
from cyclebench.schema import AnalysisMode, HealthEvent


def _confidence_from(n_episodes: int, completeness: float, regularity_cv: Optional[float]) -> str:
    if n_episodes < 3 or completeness < 0.5:
        return "low"
    if regularity_cv is not None and regularity_cv > 0.15:
        return "low"
    if n_episodes >= 4 and completeness >= 0.8:
        return "high"
    return "medium"


def detect_cyclical_pattern(
    episodes: list[HealthEvent],
    onsets: list[date],
    mode: AnalysisMode,
) -> dict:
    """Assign each episode to a cycle phase and report clustering in the dominant phase.

    Data-driven: we do not pre-declare the window; the dominant phase is discovered,
    then the observed count is compared to a uniform-distribution baseline.
    """
    if mode is None:
        raise ValueError("analysis mode is mandatory")

    dated = [e for e in episodes if e.start is not None]
    n_total = len(episodes)

    phase_of: dict[str, str] = {}
    used_future_any = False
    for e in dated:
        pos = assign_cycle_position(e.start, onsets, mode)
        if pos.phase is not None:
            phase_of[e.event_id] = pos.phase
        used_future_any = used_future_any or pos.used_future_data

    n_aligned = len(phase_of)
    completeness = (n_aligned / n_total) if n_total else 0.0

    counts: dict[str, int] = {}
    for ph in phase_of.values():
        counts[ph] = counts.get(ph, 0) + 1

    if not counts:
        return {
            "n_episodes": n_total,
            "n_aligned": 0,
            "dominant_phase": None,
            "n_in_window": 0,
            "baseline_expected": None,
            "relative_frequency": None,
            "n_cycles_covered": 0,
            "completeness": round(completeness, 3),
            "confidence": "low",
            "used_future_data": used_future_any,
            "note": "no episodes could be aligned to a cycle",
        }

    dominant_phase = max(counts, key=counts.get)
    n_in_window = counts[dominant_phase]

    # Uniform baseline: expected episodes in this phase if spread evenly across the
    # cycle, weighted by the phase's fractional width.
    phase_widths = {
        "menstrual": 0.15,
        "follicular": 0.30,
        "periovulatory": 0.15,
        "luteal": 0.40,
    }
    width = phase_widths.get(dominant_phase, 0.25)
    baseline_expected = n_aligned * width
    relative_frequency = (
        (n_in_window / baseline_expected) if baseline_expected > 0 else None
    )

    n_cycles = max(0, len(sorted(set(onsets))) - 1)
    reg_cv = cycle_regularity(onsets)
    confidence = _confidence_from(n_aligned, completeness, reg_cv)

    return {
        "n_episodes": n_total,
        "n_aligned": n_aligned,
        "dominant_phase": dominant_phase,
        "n_in_window": n_in_window,
        "phase_counts": counts,
        "baseline_expected": round(baseline_expected, 2),
        "relative_frequency": round(relative_frequency, 2) if relative_frequency else None,
        "n_cycles_covered": n_cycles,
        "completeness": round(completeness, 3),
        "regularity_cv": round(reg_cv, 3) if reg_cv is not None else None,
        "confidence": confidence,
        "used_future_data": used_future_any,
        "note": (
            f"{n_in_window} of {n_total} episodes fell in the {dominant_phase} window"
        ),
    }


def detect_change_after_event(
    episodes: list[HealthEvent],
    change_date: date,
    window_days: int = 90,
) -> dict:
    """Compare episode frequency in the `window_days` before vs after a change date.

    Reports rates per 30 days; no causal claim is implied by this function.
    """
    dated = sorted((e for e in episodes if e.start is not None), key=lambda e: e.start)
    before = [e for e in dated if e.start < change_date]
    after = [e for e in dated if e.start >= change_date]

    def rate(evs: list[HealthEvent]) -> Optional[float]:
        if not evs:
            return 0.0
        span = max((max(e.start for e in evs) - min(e.start for e in evs)).days, window_days)
        return round(len(evs) / span * 30.0, 3)

    r_before = rate(before)
    r_after = rate(after)
    direction = "increase" if (r_after or 0) > (r_before or 0) else (
        "decrease" if (r_after or 0) < (r_before or 0) else "no_change"
    )
    return {
        "change_date": change_date.isoformat(),
        "n_before": len(before),
        "n_after": len(after),
        "rate_before_per_30d": r_before,
        "rate_after_per_30d": r_after,
        "direction": direction,
        "confidence": "low" if (len(before) + len(after)) < 4 else "medium",
        "note": (
            f"episode rate went from {r_before}/30d before to {r_after}/30d after "
            f"the change ({direction})"
        ),
    }
