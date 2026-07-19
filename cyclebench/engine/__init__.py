"""CycleBench deterministic engine.

Every function here is deterministic and inspectable. No LLM, no randomness in the
analysis path. The engine turns a Case into a set of provenance-linked Findings and,
finally, a DoctorBrief.
"""

from cyclebench.engine.timeline import TimelineEntry, compile_timeline
from cyclebench.engine.cycle import assign_cycle_position, estimate_cycle_lengths
from cyclebench.engine.patterns import detect_change_after_event, detect_cyclical_pattern
from cyclebench.engine.confounders import detect_confounders
from cyclebench.engine.missing import detect_missing_information
from cyclebench.engine.pipeline import compile_case

__all__ = [
    "TimelineEntry",
    "compile_timeline",
    "assign_cycle_position",
    "estimate_cycle_lengths",
    "detect_cyclical_pattern",
    "detect_change_after_event",
    "detect_confounders",
    "detect_missing_information",
    "compile_case",
]
