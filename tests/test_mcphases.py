"""mcPHASES validation tests.

These SKIP automatically when the restricted dataset is not present (e.g. in the
published repo), so the suite stays green without redistributing the data.
"""

import pytest

from cyclebench.adapters.mcphases_validate import MCPHASES, _is_bleeding, _to_num, run_validation


def test_ordinal_mapping():
    assert _to_num("Not at all") == 0
    assert _to_num("Very High") == 5
    assert _to_num("3") == 3
    assert _to_num(None) is None


def test_bleeding_detection():
    assert _is_bleeding("Heavy") is True
    assert _is_bleeding("Not at all") is False
    assert _is_bleeding(float("nan")) is False


@pytest.mark.skipif(not MCPHASES.exists(), reason="restricted mcPHASES data not present")
def test_validation_runs_and_shapes():
    out = run_validation()
    assert out["n_participants"] == 42
    hc = out["symptom_phase_clustering"]["headaches"]
    assert hc["n_episodes"] > 0
    # engine agreement should beat 4-class chance (0.25)
    agr = out["engine_cycle_alignment_validation"]["agreement"]
    assert agr is None or agr > 0.25
