"""Golden snapshot for the Sarah demo — the acceptance test for Tier A."""

from cyclebench.engine import compile_case
from cyclebench.fixtures import build_sarah_case
from cyclebench.safety import assert_brief_safe, find_violations
from cyclebench.schema import AnalysisMode, EstablishmentClass


def _compile(mode):
    return compile_case(build_sarah_case(), mode)


def test_retrospective_finds_four_of_five_in_window():
    res = _compile(AnalysisMode.retrospective)
    cyc = next(f for f in res.findings if f.finding_id == "F_cyclical")
    assert cyc.metrics["n_in_window"] == 4
    assert cyc.metrics["n_episodes"] == 5
    assert cyc.metrics["dominant_phase"] == "luteal"


def test_causal_is_more_conservative_than_retrospective():
    retro = _compile(AnalysisMode.retrospective)
    causal = _compile(AnalysisMode.causal)
    r = next(f for f in retro.findings if f.finding_id == "F_cyclical").metrics["n_in_window"]
    c = next(f for f in causal.findings if f.finding_id == "F_cyclical").metrics["n_in_window"]
    # Causal cannot peek at the next onset, so it should not over-count.
    assert c <= r


def test_sleep_confounder_present():
    res = _compile(AnalysisMode.retrospective)
    sleep = next(f for f in res.findings if f.finding_id == "F_sleep_confounder")
    assert sleep.metrics["n_overlapping_episodes"] == 3


def test_missing_information_includes_key_fields():
    res = _compile(AnalysisMode.retrospective)
    fields = {f.metrics.get("field") for f in res.findings
              if f.establishment == EstablishmentClass.missing}
    assert "contraceptive_formulation" in fields
    assert "menstrual_onset_date" in fields


def test_every_asserting_finding_has_provenance():
    res = _compile(AnalysisMode.retrospective)
    for f in res.findings:
        if f.establishment in (EstablishmentClass.established, EstablishmentClass.possible):
            assert f.supporting_event_ids or f.source_ids, f.finding_id


def test_doctor_brief_is_safe_and_complete():
    res = _compile(AnalysisMode.retrospective)
    brief = res.brief
    assert_brief_safe(brief)  # raises if unsafe
    assert brief.opening_statement
    assert len(brief.strongest_findings) >= 1
    assert len(brief.unresolved_questions) == 3
    # the four establishment buckets are all populated for Sarah
    assert brief.established and brief.possible and brief.not_established and brief.missing
    # whole brief contains no forbidden affirmative language
    for txt in [brief.opening_statement] + brief.strongest_findings + brief.possible:
        assert not find_violations(txt)


def test_brief_declares_mode():
    assert _compile(AnalysisMode.causal).brief.analysis_mode == AnalysisMode.causal
