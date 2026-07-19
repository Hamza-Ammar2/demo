"""Build a CycleBench Case from lightweight intake answers.

Deterministic: turns structured chat answers (last period, symptoms, contraception,
sleep) into schema-valid HealthEvents. The engine then does all analysis. No LLM here —
the optional LLM lives in cyclebench.llm and only phrases language, never invents events.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

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

INTAKE_SRC = "self_report_intake"

# Symptom vocabulary the intake understands -> canonical label.
SYMPTOM_LABELS = {
    "headache": "headache",
    "migraine": "severe migraine",
    "cramps": "cramps",
    "fatigue": "fatigue",
    "brain_fog": "brain fog",
    "mood": "mood swings",
    "bloating": "bloating",
    "sore_breasts": "breast tenderness",
    "nausea": "nausea",
    "pelvic_pain": "pelvic pain",
}

SEVERITY = {"mild": 3.0, "moderate": 6.0, "severe": 8.0}


def _today() -> date:
    return date.today()


def build_case_from_intake(intake: dict, subject_id: str = "self") -> Case:
    """Construct a Case from an intake dict.

    Expected (all optional) keys:
      age_range: str                      e.g. "30-39"
      contraception_status: str           "none" | "on_stable" | "changed"
      contraception_formulation: str|None
      last_period_days_ago: int|None      days since day-1 of last period
      cycle_length_days: int|None         typical cycle length (default 28)
      prior_period_days_ago: int|None     day-1 of the period before last
      sleep_quality: str|None             "ok" | "rough" | "bad"
      symptoms: list[{type, severity, days_ago?}]
      free_text: str|None                 (used by LLM layer only, not here)
    """
    today = _today()
    events: list[HealthEvent] = []
    cycles: list[CycleContext] = []
    eid = 0

    def nid(prefix: str) -> str:
        nonlocal eid
        eid += 1
        return f"{prefix}{eid}"

    subject = SubjectProfile(
        subject_id=subject_id,
        age_range=intake.get("age_range"),
        life_stage=intake.get("life_stage"),
        contraception_status=intake.get("contraception_status"),
        contraception_formulation=intake.get("contraception_formulation"),
        source_quality=Certainty.low,  # self-report intake
    )

    # --- menstrual onsets from "days ago" answers ---
    onsets: list[date] = []
    last_days = intake.get("last_period_days_ago")
    cyc_len = int(intake.get("cycle_length_days") or 28)
    if isinstance(last_days, (int, float)):
        last_onset = today - timedelta(days=int(last_days))
        onsets.append(last_onset)
        # reconstruct a couple of prior onsets so the engine can align a window
        prior = intake.get("prior_period_days_ago")
        if isinstance(prior, (int, float)):
            onsets.append(today - timedelta(days=int(prior)))
        else:
            onsets.append(last_onset - timedelta(days=cyc_len))
            onsets.append(last_onset - timedelta(days=2 * cyc_len))

    onsets = sorted(set(onsets))
    for od in onsets:
        events.append(HealthEvent(
            event_id=nid("o"), subject_id=subject_id,
            event_type=EventType.menstrual_onset, label="period onset",
            start=od, date_precision=DatePrecision.day,
            certainty=Certainty.low, evidence_class=EvidenceClass.patient_reported,
            source_id=INTAKE_SRC,
        ))
        cycles.append(CycleContext(
            cycle_id=nid("cyc"), subject_id=subject_id, period_onset=od,
            analysis_mode=AnalysisMode.retrospective, anchor_source=INTAKE_SRC,
        ))

    # --- symptoms ---
    for s in intake.get("symptoms", []) or []:
        stype = s.get("type")
        label = SYMPTOM_LABELS.get(stype, stype or "symptom")
        sev = SEVERITY.get(s.get("severity", "moderate"), 6.0)
        days_ago = s.get("days_ago")
        when = today - timedelta(days=int(days_ago)) if isinstance(days_ago, (int, float)) else today
        prec = DatePrecision.day if isinstance(days_ago, (int, float)) else DatePrecision.approximate
        events.append(HealthEvent(
            event_id=nid("s"), subject_id=subject_id, event_type=EventType.symptom,
            label=label, start=when, date_precision=prec, severity=sev,
            certainty=Certainty.low, evidence_class=EvidenceClass.patient_reported,
            source_id=INTAKE_SRC,
        ))

    # --- sleep signal (single recent night as a coarse flag) ---
    sq = intake.get("sleep_quality")
    if sq in ("rough", "bad"):
        hours = 5.0 if sq == "bad" else 6.0
        events.append(HealthEvent(
            event_id=nid("sl"), subject_id=subject_id,
            event_type=EventType.sleep_measurement, label="sleep hours",
            start=today, date_precision=DatePrecision.approximate,
            value=hours, unit="hours", certainty=Certainty.low,
            evidence_class=EvidenceClass.patient_reported, source_id=INTAKE_SRC,
        ))

    # --- contraception change ---
    if intake.get("contraception_status") == "changed":
        cc_days = intake.get("contraception_changed_days_ago", 60)
        events.append(HealthEvent(
            event_id=nid("c"), subject_id=subject_id,
            event_type=EventType.contraception_changed,
            label="contraception changed",
            start=today - timedelta(days=int(cc_days)),
            date_precision=DatePrecision.approximate, certainty=Certainty.low,
            evidence_class=EvidenceClass.patient_reported, source_id=INTAKE_SRC,
        ))

    sources = [SourceReference(
        source_id=INTAKE_SRC, source_type=SourceType.text,
        excerpt="Self-reported intake (web questionnaire). Not verified clinical data.",
        confidence=Certainty.low,
    )]

    return Case(subject=subject, events=events, cycles=cycles, sources=sources)


# --- menopause-stage feature mapping (intake -> the SWAN model's feature space) ---

_AGE_MID = {"under-20": 18, "20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55}
_SLEEP_TO_DISTURBANCE = {"ok": 0.0, "rough": 2.0, "bad": 3.0}
_CYCLE_TO_IRREG = {"regular": 0.0, "irregular": 1.0, "very_irregular": 2.0, "none": 2.0}


def intake_to_menopause_features(intake: dict) -> dict:
    """Map self-report answers onto the menopause-stage model's feature names.

    Only fields a person can plausibly answer are set; unknown labs (FSH/E2/SHBG/BMI)
    are left absent so the model's imputer fills the cohort median. Values are coarse
    proxies, so the output is framed as an *estimate*, never a diagnosis.
    """
    feats: dict[str, float] = {}

    age = _AGE_MID.get(intake.get("age_range"))
    if age is not None:
        feats["age_years"] = float(age)

    # amenorrhea (months) from time since last period / no-periods answer
    lp = intake.get("last_period_days_ago")
    if intake.get("cycle_regularity") == "none":
        feats["amenorrhea_months"] = 12.0
    elif isinstance(lp, (int, float)):
        feats["amenorrhea_months"] = round(max(0.0, (float(lp) - 35.0) / 30.0), 2)

    if intake.get("sleep_quality") in _SLEEP_TO_DISTURBANCE:
        feats["sleep_disturbance"] = _SLEEP_TO_DISTURBANCE[intake["sleep_quality"]]
    if intake.get("cycle_regularity") in _CYCLE_TO_IRREG:
        feats["cycle_irregularity"] = _CYCLE_TO_IRREG[intake["cycle_regularity"]]

    if isinstance(intake.get("hot_flash_freq"), (int, float)):
        feats["hot_flash_freq"] = float(intake["hot_flash_freq"])
    if isinstance(intake.get("night_sweat_freq"), (int, float)):
        feats["night_sweat_freq"] = float(intake["night_sweat_freq"])

    # coarse FSH proxy only if the user reports bloodwork
    bw = intake.get("bloodwork")
    if bw == "fsh_high":
        feats["fsh_miu_ml"] = 45.0
    elif bw == "fsh_normal":
        feats["fsh_miu_ml"] = 8.0

    return feats


def menopause_relevant(intake: dict) -> bool:
    """Only estimate menopause stage when it's contextually appropriate.

    Avoids raising menopause with, e.g., a 22-year-old. Triggers on midlife age,
    vasomotor symptoms, or prolonged amenorrhea.
    """
    age = _AGE_MID.get(intake.get("age_range"), 0)
    if age >= 40:
        return True
    if (intake.get("hot_flash_freq") or 0) >= 2 or (intake.get("night_sweat_freq") or 0) >= 2:
        return True
    if intake.get("cycle_regularity") == "none":
        return True
    lp = intake.get("last_period_days_ago")
    if isinstance(lp, (int, float)) and lp >= 90:
        return True
    return False
