import pytest

from cyclebench.safety import SafetyViolation, assert_safe, find_violations


def test_flags_diagnostic_assertion():
    assert find_violations("You have PCOS.")
    assert find_violations("This is caused by your hormones.")
    assert find_violations("We recommend stopping your medication.")
    assert find_violations("You should stop the pill.")


def test_allows_negated_and_safe_language():
    assert not find_violations("This does not diagnose or establish causation.")
    assert not find_violations("No causal link is established.")
    assert not find_violations("Four of five episodes occurred in a comparable window.")
    assert not find_violations("A repeated temporal association was observed.")


def test_assert_safe_raises():
    with pytest.raises(SafetyViolation):
        assert_safe("You are suffering from endometriosis.")
