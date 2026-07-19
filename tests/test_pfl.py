"""Unit tests for fed-branch personalized FL helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from cyclebench.model.pfl import (
    FEATURE_COLS,
    GLOBAL_MODEL_PATH,
    average_state_dicts,
    build_sequential_dataset,
    run_pfl_inference,
)


def test_average_state_dicts_means_tensors():
    import torch

    a = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([1.0])}
    b = {"w": torch.tensor([2.0, 4.0]), "b": torch.tensor([3.0])}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(avg["b"], torch.tensor([2.0]))
    # inputs unchanged
    assert torch.allclose(a["w"], torch.tensor([0.0, 2.0]))


def test_average_state_dicts_empty_raises():
    with pytest.raises(ValueError):
        average_state_dicts([])


def test_build_sequential_dataset_window():
    import pandas as pd

    rows = []
    for i in range(7):
        row = {c: float(i) for c in FEATURE_COLS}
        row["phase"] = ["Menstrual", "Follicular", "Fertility", "Luteal"][i % 4]
        rows.append(row)
    df = pd.DataFrame(rows)
    X, y = build_sequential_dataset(df, w=5)
    assert X is not None and y is not None
    assert X.shape == (3, 5, len(FEATURE_COLS))
    assert y.shape == (3,)


def test_build_sequential_dataset_too_short():
    import pandas as pd

    df = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS} | {"phase": "Luteal"}] * 3)
    X, y = build_sequential_dataset(df, w=5)
    assert X is None and y is None


@pytest.mark.skipif(not GLOBAL_MODEL_PATH.exists(), reason="global_pfl_model.pt missing")
def test_pfl_inference_pad_flag(tmp_path, monkeypatch):
    # Isolate from any real local logs
    import cyclebench.model.pfl as pfl

    monkeypatch.setattr(pfl, "LOCAL_DATA_PATH", tmp_path / "local.csv")
    monkeypatch.setattr(pfl, "LOCAL_MODEL_PATH", tmp_path / "local.pt")
    # Keep global checkpoint; allow writing meta next to models or tmp
    meta = tmp_path / "meta.npz"
    monkeypatch.setattr(pfl, "GLOBAL_META_PATH", meta)
    # Provide trivial norms so we do not require mcPHASES for this unit test
    mean = np.zeros(len(FEATURE_COLS), dtype=np.float32)
    std = np.ones(len(FEATURE_COLS), dtype=np.float32)
    np.savez(meta, mean=mean, std=std)

    feats = {c: 1.0 for c in FEATURE_COLS}
    pred = run_pfl_inference(feats)
    assert pred["model"] == "Personalized_GRU_FedPer"
    assert pred["predicted_state"] in {"Menstrual", "Follicular", "Fertility", "Luteal"}
    assert pred["sequence_padded"] is True
    assert "probabilities" in pred
