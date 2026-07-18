"""Case compilation pipeline.

Runs the deterministic engine end-to-end:
  events -> timeline -> cycle alignment -> pattern detection -> confounders
         -> missing-info -> provenance-linked Findings -> DoctorBrief

The `mode` argument is mandatory and threaded through every cycle-dependent step.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from cyclebench.case import Case
from cyclebench.engine.confounders import detect_confounders
from cyclebench.engine.missing import detect_missing_information
from cyclebench.engine.patterns import detect_change_after_event, detect_cyclical_pattern
from cyclebench.engine.timeline import compile_timeline
from cyclebench.safety import assert_brief_safe, assert_safe
from cyclebench.schema import (
    AnalysisMode,
    Certainty,
    DoctorBrief,
    EstablishmentClass,
    EventType,
    Finding,
    FindingType,
)

_CERTAINTY = {"high": Certainty.high, "medium": Certainty.medium, "low": Certainty.low}


def _severe_symptom_episodes(case: Case, min_severity: float = 7.0):
    """Symptom events that are 'severe' (severity >= threshold, or explicitly labeled)."""
    out = []
    for e in case.events_of(EventType.symptom):
        if e.severity is not None and e.severity >= min_severity:
            out.append(e)
        elif e.severity is None and e.label and "severe" in e.label.lower():
            out.append(e)
    return out


class CompileResult:
    def __init__(self, timeline, findings: list[Finding], brief: DoctorBrief):
        self.timeline = timeline
        self.findings = findings
        self.brief = brief


def compile_case(case: Case, mode: AnalysisMode) -> CompileResult:
    """Compile a Case into a timeline, findings, and a DoctorBrief under `mode`."""
    if mode is None:
        raise ValueError("analysis mode is mandatory (retrospective|causal)")

    timeline = compile_timeline(case.events)
    onsets = [e.start for e in case.events_of(EventType.menstrual_onset) if e.start]
    episodes = _severe_symptom_episodes(case)

    findings: list[Finding] = []

    # --- Confounders (computed first so they can weaken the main finding). ---
    confounders = detect_confounders(episodes, case.events)
    sleep_conf = next((c for c in confounders if c["type"] == "poor_sleep"), None)
    conf_event_ids = [cid for c in confounders for cid in c.get("confounder_event_ids", [])]

    # --- Finding 1: cyclical pattern. ---
    if episodes and onsets:
        pat = detect_cyclical_pattern(episodes, onsets, mode)
        if pat["dominant_phase"] and pat["n_in_window"] >= 2:
            weakened = " Reduced sleep overlapped several episodes and remains a " \
                       "possible contributor." if sleep_conf else ""
            statement = (
                f"{pat['n_in_window']} of {pat['n_episodes']} severe episodes occurred in a "
                f"comparable cycle window (relative frequency "
                f"{pat['relative_frequency']}x a uniform baseline)." + weakened
            )
            assert_safe(statement, where="finding.cyclical")
            findings.append(Finding(
                finding_id="F_cyclical",
                title="Repeated temporal association with a cycle window",
                statement=statement,
                finding_type=FindingType.temporal_association,
                supporting_event_ids=[e.event_id for e in episodes if e.start],
                confounder_event_ids=sleep_conf["confounder_event_ids"] if sleep_conf else [],
                source_ids=sorted({e.source_id for e in episodes if e.source_id}),
                strength=_CERTAINTY.get(pat["confidence"], Certainty.low),
                method="detect_cyclical_pattern/phase-fraction-binning",
                analysis_mode=mode,
                establishment=EstablishmentClass.possible,
                limitation=(
                    "Association only; small number of cycles; "
                    + ("irregular cycles reduce confidence; " if pat.get("regularity_cv") and pat["regularity_cv"] > 0.15 else "")
                    + "sleep is a possible confounder." if sleep_conf else
                    "Association only; small number of cycles."
                ),
                metrics=pat,
            ))

    # --- Finding 2: change after contraception change. ---
    contra = case.events_of(EventType.contraception_changed)
    if episodes and contra and contra[0].start:
        chg = detect_change_after_event(episodes, contra[0].start)
        if chg["direction"] == "increase":
            statement = (
                f"Severe-episode frequency was higher after the recorded contraception "
                f"change ({chg['rate_after_per_30d']}/30d) than before "
                f"({chg['rate_before_per_30d']}/30d)."
            )
            assert_safe(statement, where="finding.change_after")
            findings.append(Finding(
                finding_id="F_change_after_contraception",
                title="Symptom frequency changed after contraception change",
                statement=statement,
                finding_type=FindingType.change_after_event,
                supporting_event_ids=[e.event_id for e in episodes if e.start],
                confounder_event_ids=[contra[0].event_id],
                source_ids=sorted({e.source_id for e in episodes if e.source_id}),
                strength=_CERTAINTY.get(chg["confidence"], Certainty.low),
                method="detect_change_after_event/before-after-rate",
                analysis_mode=mode,
                establishment=EstablishmentClass.possible,
                limitation="Association only; before/after comparison, not a controlled design.",
                metrics=chg,
            ))

    # --- Finding 3: sleep confounder (as its own explicit finding). ---
    if sleep_conf:
        assert_safe(sleep_conf["statement"], where="finding.sleep")
        findings.append(Finding(
            finding_id="F_sleep_confounder",
            title="Reduced sleep overlaps severe episodes",
            statement=sleep_conf["statement"],
            finding_type=FindingType.confounder,
            supporting_event_ids=sleep_conf["episode_event_ids"],
            confounder_event_ids=sleep_conf["confounder_event_ids"],
            strength=Certainty.medium,
            method="detect_confounders/poor_sleep-overlap",
            analysis_mode=mode,
            establishment=EstablishmentClass.possible,
            limitation="Overlap is not evidence of cause; sleep and cycle effects are entangled.",
            metrics={k: v for k, v in sleep_conf.items() if k != "statement"},
        ))

    # --- Missing information -> findings (establishment = missing). ---
    missing_items = detect_missing_information(case)
    for i, m in enumerate(missing_items):
        findings.append(Finding(
            finding_id=f"F_missing_{i}",
            title=f"Missing: {m['field']}",
            statement=m["statement"],
            finding_type=FindingType.missing_information,
            method="detect_missing_information",
            analysis_mode=mode,
            establishment=EstablishmentClass.missing,
            metrics={"priority": m["priority"], "field": m["field"]},
        ))

    brief = _build_doctor_brief(case, findings, mode, missing_items)
    assert_brief_safe(brief)
    return CompileResult(timeline, findings, brief)


def _build_doctor_brief(case, findings, mode, missing_items) -> DoctorBrief:
    asserting = [f for f in findings if f.establishment == EstablishmentClass.possible]
    strongest = [f.statement for f in asserting[:3]]

    questions: list[str] = []
    for m in missing_items[:3]:
        field = m["field"]
        if field == "contraceptive_formulation":
            questions.append("What exact contraceptive formulation and dose were used, and when?")
        elif field == "medication_detail":
            questions.append("Could the timing or dose of current medication be relevant?")
        elif field == "menstrual_onset_date":
            questions.append("Can the missing or approximate period dates be confirmed?")
        else:
            questions.append(f"Can we clarify: {m['statement']}")
    while len(questions) < 3:
        questions.append("What other non-hormonal explanations should be investigated?")

    opening = _opening_statement(case, asserting, mode)
    assert_safe(opening, where="opening_statement")

    established = _established_facts(case)
    possible = [f.title for f in asserting]
    not_established = [
        "No diagnosis is implied or established for these symptoms.",
        "No causal link between hormones or contraception and the symptoms is established.",
        "No treatment effect is established.",
    ]
    missing = [m["statement"] for m in missing_items]

    return DoctorBrief(
        subject_id=case.subject.subject_id,
        analysis_mode=mode,
        opening_statement=opening,
        strongest_findings=strongest,
        unresolved_questions=questions[:3],
        established=established,
        possible=possible,
        not_established=not_established,
        missing=missing,
        finding_ids=[f.finding_id for f in findings],
    )


def _opening_statement(case, asserting, mode) -> str:
    symptom_labels = sorted({
        (e.label or e.event_type.value)
        for e in case.events_of(EventType.symptom)
    })
    symptom_phrase = ", ".join(symptom_labels[:3]) if symptom_labels else "recurring symptoms"
    lead = (
        f"Over the recorded period I have experienced {symptom_phrase}. "
    )
    if asserting:
        lead += asserting[0].statement + " "
    lead += (
        "These are observed associations, not established causes. I would like to review the "
        "timeline together and agree on what to investigate next."
    )
    return lead


def _established_facts(case) -> list[str]:
    facts: list[str] = []
    n_sym = len(case.events_of(EventType.symptom))
    if n_sym:
        facts.append(f"{n_sym} recorded symptom episode(s) with dates.")
    if case.events_of(EventType.contraception_changed):
        facts.append("A recorded contraception-change date.")
    if case.events_of(EventType.dose_changed, EventType.medication_started):
        facts.append("A recorded medication/dose-change date.")
    if case.events_of(EventType.sleep_measurement):
        facts.append("Recorded sleep measurements.")
    return facts
