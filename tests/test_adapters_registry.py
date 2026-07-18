import pytest

from cyclebench.adapters.nhanes_harmonize import PROCESSED, _age_band, harmonize
from cyclebench.adapters.registry import REGISTRY, as_markdown, can_support


def test_age_band():
    assert _age_band(34) == "28-37"
    assert _age_band(None) == "unknown"
    assert _age_band(90) == "80_plus"


def test_registry_guards_invalid_uses():
    # mcPHASES must NOT be used for menopause or hormone-level clinical prediction
    assert can_support("mcphases", "cycle-phase") is True
    assert can_support("mcphases", "menopause") is False
    assert can_support("mcphases", "hormone-level") is False
    # NHANES must NOT be treated as longitudinal
    assert can_support("nhanes", "reference ranges") is True
    assert can_support("nhanes", "within-person cycle") is False


def test_registry_flags_mcphases_not_redistributable():
    assert REGISTRY["mcphases"].redistributable is False
    assert REGISTRY["nhanes"].redistributable is True
    assert "Dataset Registry" in as_markdown()


@pytest.mark.skipif(not (PROCESSED / "P_TST.csv").exists(),
                    reason="NHANES processed CSVs not present")
def test_harmonize_produces_expected_shapes():
    out = harmonize()
    assert out["n_subjects"] > 1000
    assert out["n_hormone_events"] > out["n_subjects"]
    assert "estradiol" in out["analytes"] and "shbg" in out["analytes"]
