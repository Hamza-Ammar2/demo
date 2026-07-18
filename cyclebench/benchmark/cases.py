"""CycleBench-Bench v0.1 — documented synthetic evaluation cases.

ALL cases are SYNTHETIC and labeled as such. They are designed to test whether a
system turns fragmented longitudinal input into an accurate AND safe doctor brief,
including cases where the correct answer is "no meaningful pattern".

Each BenchCase carries the ground-truth structured events plus the expected
analytical outcome, so metrics can be computed deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from cyclebench.case import Case
from cyclebench.schema import (
    Certainty,
    DatePrecision,
    EvidenceClass,
    EventType,
    HealthEvent,
    SourceReference,
    SourceType,
    SubjectProfile,
)

SRC = "bench_fixture"


@dataclass
class BenchCase:
    case_id: str
    category: str  # positive | negative | misleading | irregular | insufficient
    narrative: str
    case: Case
    expect_cyclical_pattern: bool
    expect_confounders: set = field(default_factory=set)
    expect_missing_fields: set = field(default_factory=set)
    expect_change_after_event: bool = False


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _subject(sid, **kw):
    return SubjectProfile(subject_id=sid, age_range="30-39", life_stage="reproductive", **kw)


def _sym(sid, eid, d, label, sev):
    return HealthEvent(event_id=eid, subject_id=sid, event_type=EventType.symptom,
                       label=label, start=d, date_precision=DatePrecision.day, severity=sev,
                       evidence_class=EvidenceClass.patient_reported, source_id=SRC)


def _onset(sid, eid, d, precision=DatePrecision.day):
    return HealthEvent(event_id=eid, subject_id=sid, event_type=EventType.menstrual_onset,
                       label="period onset", start=d, date_precision=precision,
                       evidence_class=EvidenceClass.patient_reported, source_id=SRC)


def _sleep(sid, eid, d, hrs):
    return HealthEvent(event_id=eid, subject_id=sid, event_type=EventType.sleep_measurement,
                       start=d, date_precision=DatePrecision.day, value=hrs, unit="hours",
                       evidence_class=EvidenceClass.measured, source_id=SRC)


def _sources():
    return [SourceReference(source_id=SRC, source_type=SourceType.synthetic_fixture,
                            excerpt="Synthetic benchmark case. Not real patient data.",
                            confidence=Certainty.medium)]


def _regular_onsets(sid, start: date, n=6, length=28):
    return [_onset(sid, f"{sid}_o{i}", start + timedelta(days=length * i)) for i in range(n)]


def _case(sid, subject, events):
    return Case(subject=subject, events=events, sources=_sources())


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # 1. positive: luteal clustering + sleep confounder + missing formulation
    sid = "c01"
    onsets = _regular_onsets(sid, date(2024, 1, 3))
    epis = [_sym(sid, f"{sid}_m{i}", o.start + timedelta(days=22), "severe migraine", 8)
            for i, o in enumerate(onsets[:4])]
    epis.append(_sym(sid, f"{sid}_m4", onsets[4].start + timedelta(days=14), "severe migraine", 8))
    sleeps = [_sleep(sid, f"{sid}_s{i}", epis[i].start, 5.0) for i in range(3)]
    contra = HealthEvent(event_id=f"{sid}_c", subject_id=sid,
                         event_type=EventType.contraception_changed, start=date(2023, 12, 20),
                         date_precision=DatePrecision.day, source_id=SRC)
    cases.append(BenchCase(
        sid, "positive",
        "Migraines keep hitting the week before my period for months; sleep is rough around then. "
        "I switched birth control late last year but don't recall the exact type.",
        _case(sid, _subject(sid, contraception_status="changed"), onsets + epis + sleeps + [contra]),
        expect_cyclical_pattern=True, expect_confounders={"poor_sleep"},
        expect_missing_fields={"contraceptive_formulation"}, expect_change_after_event=True,
    ))

    # 2. positive: perimenstrual headaches, no confounder
    sid = "c02"
    onsets = _regular_onsets(sid, date(2024, 1, 1))
    epis = [_sym(sid, f"{sid}_h{i}", o.start + timedelta(days=1), "headache", 8)
            for i, o in enumerate(onsets[:5])]
    cases.append(BenchCase(
        sid, "positive",
        "Headaches almost always land on the first day or two of my period.",
        _case(sid, _subject(sid), onsets + epis),
        expect_cyclical_pattern=True,
    ))

    # 3. negative: episodes spread evenly across the cycle
    sid = "c03"
    onsets = _regular_onsets(sid, date(2024, 1, 1))
    offsets = [2, 9, 16, 23, 6]
    epis = [_sym(sid, f"{sid}_e{i}", onsets[i].start + timedelta(days=off), "headache", 8)
            for i, off in enumerate(offsets)]
    cases.append(BenchCase(
        sid, "negative",
        "I get headaches sometimes but they seem random, no clear timing.",
        _case(sid, _subject(sid), onsets + epis),
        expect_cyclical_pattern=False,
    ))

    # 4. negative/insufficient: symptoms but no cycle onsets recorded
    sid = "c04"
    epis = [_sym(sid, f"{sid}_e{i}", date(2024, 1, 5) + timedelta(days=20 * i), "migraine", 8)
            for i in range(4)]
    cases.append(BenchCase(
        sid, "insufficient",
        "I have migraines but I never track my periods.",
        _case(sid, _subject(sid), epis),
        expect_cyclical_pattern=False,
        expect_missing_fields=set(),
    ))

    # 5. misleading: looks cyclical but every episode overlaps poor sleep
    sid = "c05"
    onsets = _regular_onsets(sid, date(2024, 1, 2))
    epis = [_sym(sid, f"{sid}_m{i}", o.start + timedelta(days=23), "severe migraine", 9)
            for i, o in enumerate(onsets[:4])]
    sleeps = [_sleep(sid, f"{sid}_s{i}", epis[i].start, 4.5) for i in range(4)]
    cases.append(BenchCase(
        sid, "misleading",
        "My migraines seem tied to my cycle, but honestly I also barely sleep those nights.",
        _case(sid, _subject(sid), onsets + epis + sleeps),
        expect_cyclical_pattern=True, expect_confounders={"poor_sleep"},
    ))

    # 6. irregular cycles: clustering but high variability -> low confidence
    sid = "c06"
    irregular = [date(2024, 1, 1), date(2024, 1, 24), date(2024, 3, 5),
                 date(2024, 3, 22), date(2024, 5, 1), date(2024, 5, 20)]
    onsets = [_onset(sid, f"{sid}_o{i}", d) for i, d in enumerate(irregular)]
    epis = [_sym(sid, f"{sid}_m{i}", onsets[i].start + timedelta(days=18), "migraine", 8)
            for i in range(4)]
    cases.append(BenchCase(
        sid, "irregular",
        "My cycles are all over the place, but migraines feel late-cycle.",
        _case(sid, _subject(sid), onsets + epis),
        # Correct behavior: with irregular cycles and few episodes, the system should
        # NOT assert a confident cyclical pattern (it must withhold, not over-claim).
        expect_cyclical_pattern=False,
    ))

    # 7. insufficient: a single episode and a single onset
    sid = "c07"
    ev = [_onset(sid, f"{sid}_o0", date(2024, 1, 1)),
          _sym(sid, f"{sid}_m0", date(2024, 1, 20), "migraine", 8)]
    cases.append(BenchCase(
        sid, "insufficient",
        "I had one bad migraine last month.",
        _case(sid, _subject(sid), ev),
        expect_cyclical_pattern=False,
    ))

    # 8. change-after-medication: episodes increase after a dose change
    sid = "c08"
    onsets = _regular_onsets(sid, date(2024, 1, 1))
    before = [_sym(sid, f"{sid}_b0", date(2024, 1, 10), "headache", 8)]
    # After-episodes deliberately spread across phases so there is a frequency change
    # but NOT a cyclical clustering signal.
    after_dates = [date(2024, 3, 1), date(2024, 3, 12), date(2024, 4, 18), date(2024, 5, 5)]
    after = [_sym(sid, f"{sid}_a{i}", d, "headache", 8) for i, d in enumerate(after_dates)]
    contra = HealthEvent(event_id=f"{sid}_c", subject_id=sid,
                         event_type=EventType.contraception_changed, start=date(2024, 2, 20),
                         date_precision=DatePrecision.day, source_id=SRC)
    cases.append(BenchCase(
        sid, "positive",
        "After I changed contraception in late February, the headaches got much more frequent.",
        _case(sid, _subject(sid, contraception_status="changed"), onsets + before + after + [contra]),
        expect_cyclical_pattern=False, expect_change_after_event=True,
        expect_missing_fields={"contraceptive_formulation"},
    ))

    # 9. negative: stable low-grade symptoms, no escalation, no clustering
    sid = "c09"
    onsets = _regular_onsets(sid, date(2024, 1, 1))
    epis = [_sym(sid, f"{sid}_e{i}", onsets[i].start + timedelta(days=[3, 11, 19, 25][i]), "fatigue", 8)
            for i in range(4)]
    cases.append(BenchCase(
        sid, "negative",
        "I'm tired a lot but it doesn't track with anything obvious.",
        _case(sid, _subject(sid), onsets + epis),
        expect_cyclical_pattern=False,
    ))

    # 10. positive with contradicting evidence: mostly luteal, some outside
    sid = "c10"
    onsets = _regular_onsets(sid, date(2024, 1, 1))
    epis = [_sym(sid, f"{sid}_m{i}", onsets[i].start + timedelta(days=24), "severe migraine", 8)
            for i in range(3)]
    epis += [_sym(sid, f"{sid}_x{i}", onsets[i].start + timedelta(days=6), "severe migraine", 8)
             for i in range(2)]
    cases.append(BenchCase(
        sid, "positive",
        "Most of my worst migraines are late cycle, though a couple hit early too.",
        _case(sid, _subject(sid), onsets + epis),
        expect_cyclical_pattern=True,
    ))

    return cases


BENCH_CASES = build_cases()
