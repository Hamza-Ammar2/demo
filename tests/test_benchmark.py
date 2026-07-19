from cyclebench.benchmark.baselines import engine_analysis, naive_summary
from cyclebench.benchmark.cases import BENCH_CASES
from cyclebench.benchmark.runner import evaluate


def test_cases_are_labeled_and_balanced():
    cats = {c.category for c in BENCH_CASES}
    assert {"positive", "negative", "misleading", "irregular", "insufficient"} <= cats
    # at least a few no-pattern cases
    assert sum(1 for c in BENCH_CASES if not c.expect_cyclical_pattern) >= 4


def test_engine_matches_ground_truth_on_every_case():
    for bc in BENCH_CASES:
        b = engine_analysis(bc.case)
        assert b["claims_pattern"] == bc.expect_cyclical_pattern, bc.case_id


def test_engine_reproducible():
    a = evaluate()["summary"]
    b = evaluate()["summary"]
    assert a == b


def test_engine_beats_naive_and_is_safe():
    out = evaluate()["summary"]
    assert out["pattern_detection_accuracy"]["path_B_cyclebench"] >= \
        out["pattern_detection_accuracy"]["path_A_naive"]
    assert out["false_pattern_rate"]["path_B_cyclebench"] == 0.0
    assert out["provenance_coverage_B"] == 1.0
    assert out["unsupported_claim_count"]["path_B_cyclebench"] == 0
    assert out["safety_violation_count"]["path_B_cyclebench"] == 0


def test_naive_makes_unsupported_claims():
    # Confirms the contrast is real: the naive baseline over-claims and is unsafe.
    out = evaluate()["summary"]
    assert out["false_pattern_rate"]["path_A_naive"] > 0.5
    assert out["safety_violation_count"]["path_A_naive"] > 0
