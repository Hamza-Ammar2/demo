import pytest

from cyclebench.audit import (
    AuditError,
    _honest_split,
    _leaking_split,
    audit_all,
    audit_prediction_split,
    run_demo_audit,
)
from cyclebench.engine import compile_case
from cyclebench.fixtures import build_sarah_case
from cyclebench.schema import AnalysisMode


def _sarah_findings():
    return compile_case(build_sarah_case(), AnalysisMode.retrospective).findings


def test_honest_split_passes():
    assert audit_prediction_split(_honest_split()).passed


def test_leaking_split_fails():
    res = audit_prediction_split(_leaking_split())
    assert not res.passed
    failed_ids = {c["id"] for c in res.checks if not c["passed"]}
    # participant overlap(1), cycle spanning(2), causal order(3), future onset(4),
    # train-only norm(6), target leak(7) should all fire.
    for expected in (1, 2, 3, 4, 6, 7):
        assert expected in failed_ids


def test_findings_provenance_and_language_pass():
    res = audit_all(_sarah_findings(), None)
    assert res.passed


def test_strict_mode_raises_on_leak():
    with pytest.raises(AuditError):
        audit_all(_sarah_findings(), _leaking_split(), strict=True)


def test_demo_audit_returns_true():
    assert run_demo_audit() is True
