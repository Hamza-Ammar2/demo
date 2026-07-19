"""Personalized sequence research signal (FedPer / multi-symptom GRU).

User-facing name: **Personalized sequence (research)**
Internal task id: ``sequence_research``

This is Track-02 research infrastructure, not a clinical diagnostic model.
It estimates short-horizon *cramp-pattern likelihood* from a 5-day window of
hormones + symptoms when that history is present. Otherwise the soft read can
still surface the latest cohort benchmark from ``results/pfl_multi_symptom.json``.

Torch is optional — if missing, only the cohort benchmark card is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cyclebench.safety import assert_safe

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = MODELS_DIR / "sequence_research_v0.1.pt"
METRICS_PATH = RESULTS_DIR / "pfl_multi_symptom.json"

WINDOW = 5
HORM_COLS = ["lh", "estrogen", "pdg"]
SYMP_COLS = [
    "headaches", "sorebreasts", "fatigue", "sleepissue",
    "moodswing", "stress", "bloating",
]
FEATURE_COLS = HORM_COLS + SYMP_COLS

SYMPTOM_MAP = {
    "Not at all": 0,
    "Very Low/Little": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5,
}

# Intake severity labels → ordinal (best-effort)
_INTAKE_SEV = {
    "none": 0, "mild": 2, "moderate": 3, "severe": 4, "very_severe": 5,
    "low": 2, "high": 4,
}

DISPLAY_NAME = "Personalized sequence (research)"
TASK_ID = "sequence_research"


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _build_modules():
    torch, nn = _torch()

    class GRUProjection(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=16, output_dim=8):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.ln = nn.LayerNorm(output_dim)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.ln(self.fc(out[:, -1, :]))

    class SharedEncoder(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class DecisionHead(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x):
            return self.net(x)

    class PersonalizedClientModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=16, latent_dim=8):
            super().__init__()
            self.proj = GRUProjection(input_dim, hidden_dim, latent_dim)
            self.encoder = SharedEncoder(latent_dim)
            self.head = DecisionHead(latent_dim)

        def forward(self, x):
            return self.head(self.encoder(self.proj(x)))

    return PersonalizedClientModel


def save_deployable_checkpoint(
    model_state: dict,
    mean: np.ndarray,
    std: np.ndarray,
    meta: dict | None = None,
) -> Path:
    """Persist a FedPer-derived deployable population prior for soft-read inference."""
    torch, _ = _torch()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model_state,
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "feature_cols": FEATURE_COLS,
        "window": WINDOW,
        "meta": meta or {},
    }
    torch.save(payload, CHECKPOINT)
    return CHECKPOINT


def load_cohort_benchmark() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def _window_from_intake(intake: dict) -> np.ndarray | None:
    """Return shape (1, 5, 10) if intake includes a usable sequence_days list."""
    days = intake.get("sequence_days")
    if not isinstance(days, list) or len(days) < WINDOW:
        return None
    rows = []
    for day in days[-WINDOW:]:
        if not isinstance(day, dict):
            return None
        row = []
        for col in FEATURE_COLS:
            v = day.get(col, 0.0)
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(0.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)[None, :, :]


def predict_from_window(window: np.ndarray) -> dict[str, Any]:
    """Run deployable checkpoint on (1, 5, 10) window."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"missing {CHECKPOINT} — run make pfl-smoke first")
    torch, _ = _torch()
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    mean = np.asarray(bundle["mean"], dtype=np.float32)
    std = np.asarray(bundle["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    x = (window.reshape(-1, len(FEATURE_COLS)) - mean) / std
    x = x.reshape(window.shape)

    Model = _build_modules()
    model = Model()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(x, dtype=torch.float32))
        prob = float(torch.sigmoid(logit).item())

    return {
        "task": TASK_ID,
        "display_name": DISPLAY_NAME,
        "mode": "personal_estimate",
        "cramp_pattern_probability": round(prob, 4),
        "window_days": WINDOW,
        "features_used": FEATURE_COLS,
        "model": "sequence_research_v0.1",
        "method": "FedPer-style GRU (deployable population prior)",
    }


def build_soft_read_signal(intake: dict | None = None) -> dict[str, Any] | None:
    """Signal for the soft read: personal estimate and/or cohort research context.

    Returns None only when neither checkpoint metrics nor a personal window exist.
    """
    intake = intake or {}
    bench = load_cohort_benchmark()
    personal: dict[str, Any] | None = None

    window = _window_from_intake(intake)
    if window is not None and CHECKPOINT.exists():
        try:
            personal = predict_from_window(window)
        except Exception:
            personal = None

    if personal is None and bench is None:
        return None

    # Prefer a personal estimate when available; always attach cohort context if present.
    pfl = (bench or {}).get("metrics", {}).get("Personalized FL", {})
    fedavg = (bench or {}).get("metrics", {}).get("Centralized/FedAvg", {})
    local = (bench or {}).get("metrics", {}).get("Local Only", {})

    if personal is not None:
        p = personal["cramp_pattern_probability"]
        statement = (
            f"Personalized sequence (research): multi-day hormone and symptom pattern "
            f"suggests elevated short-horizon cramp-pattern likelihood (P≈{p:.2f}). "
            f"Association-only research signal from a privacy-preserving FedPer-style model — "
            f"not a diagnosis."
        )
        mode = "personal_estimate"
        detail = f"P≈{p:.2f} · {WINDOW}-day window"
    else:
        # Cohort context — honest when chip intake has no Mira/wearable sequence
        f1 = pfl.get("f1")
        f1_s = f"{f1:.2f}" if isinstance(f1, (int, float)) else "n/a"
        statement = (
            "Personalized sequence (research): a FedPer-style model trained across patients "
            "without pooling raw records. On the latest multi-symptom mcPHASES simulation, "
            f"personalized FL reached F1≈{f1_s} for short-horizon cramp patterns. "
            "Your intake has no multi-day hormone sequence yet — this is cohort research "
            "context, not a personal prediction."
        )
        mode = "cohort_benchmark"
        detail = (
            f"Research F1≈{f1_s} (pFL) · "
            f"n={bench.get('n_clients', '?')} clients · "
            f"{bench.get('rounds', '?')} rounds"
            if bench else "Research module available"
        )

    assert_safe(statement, where="sequence_research")

    # Mention cramps in intake as relevance, not diagnosis
    has_cramps = any(
        (s.get("type") == "cramps") for s in (intake.get("symptoms") or [])
    )

    return {
        "task": TASK_ID,
        "display_name": DISPLAY_NAME,
        "mode": mode,
        "detail": detail,
        "statement": statement,
        "probability": personal.get("cramp_pattern_probability") if personal else None,
        "cohort": {
            "personalized_fl": pfl,
            "fedavg": fedavg,
            "local_only": local,
            "n_clients": (bench or {}).get("n_clients"),
            "rounds": (bench or {}).get("rounds"),
            "honesty_note": (bench or {}).get("honesty_note"),
        } if bench else None,
        "intake_mentions_cramps": has_cramps,
        "checkpoint_ready": CHECKPOINT.exists(),
        "needs_for_personal_estimate": (
            "Provide intake.sequence_days: list of ≥5 daily dicts with "
            f"keys {FEATURE_COLS}."
        ),
    }
