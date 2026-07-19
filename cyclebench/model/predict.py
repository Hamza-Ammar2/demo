"""Inference + schema Finding wrappers for Layer 02 models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from cyclebench.model.common import MODELS_DIR, explain_instance, load_bundle
from cyclebench.schema import (
    AnalysisMode,
    Certainty,
    EstablishmentClass,
    EvidenceClass,
    Finding,
    FindingType,
)


def _default_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"


def _use_pfl() -> bool:
    """Prefer FedPer GRU phase model when torch + checkpoint are available.

    Set CYCLEBENCH_USE_PFL=0 to force the sklearn hormonal-state joblib.
    """
    return os.environ.get("CYCLEBENCH_USE_PFL", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def predict_hormonal_state(
    features: dict[str, float],
    bundle_path: Optional[Path] = None,
    *,
    log_local: bool = True,
) -> dict:
    """Phase estimate: pFL GRU when available, else sklearn mcPHASES joblib."""
    path = bundle_path or _default_path("hormonal_state_v0.1")

    def _sklearn() -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint {path}; run `make train-models`")
        bundle = load_bundle(path)
        x = np.array([features.get(f, np.nan) for f in bundle.feature_names], dtype=float)
        proba = bundle.pipeline.predict_proba(x.reshape(1, -1))[0]
        pred_i = int(np.argmax(proba))
        label = str(bundle.classes[pred_i])
        expl = explain_instance(bundle.pipeline, bundle.feature_names, x, top_k=5)
        return {
            "task": bundle.task,
            "predicted_state": label,
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(bundle.classes, proba)},
            "confidence": round(float(proba[pred_i]), 4),
            "explanation": expl,
            "model": bundle.name,
        }

    pred: dict | None = None
    if _use_pfl():
        try:
            from cyclebench.model.pfl import run_pfl_inference
            pred = run_pfl_inference(features)
        except Exception:
            pred = None

    if pred is None:
        pred = _sklearn()

    if log_local:
        try:
            from cyclebench.model.pfl import save_local_log
            save_local_log(features, str(pred["predicted_state"]))
        except Exception:
            pass

    return pred


def predict_menopause_stage(
    features: dict[str, float],
    bundle_path: Optional[Path] = None,
) -> dict:
    path = bundle_path or _default_path("menopause_stage_v0.1")
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint {path}; run `make train-models`")
    bundle = load_bundle(path)
    x = np.array([features.get(f, np.nan) for f in bundle.feature_names], dtype=float)
    proba = bundle.pipeline.predict_proba(x.reshape(1, -1))[0]
    pred_i = int(np.argmax(proba))
    label = str(bundle.classes[pred_i])
    expl = explain_instance(bundle.pipeline, bundle.feature_names, x, top_k=5)
    return {
        "task": bundle.task,
        "predicted_stage": label,
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(bundle.classes, proba)},
        "confidence": round(float(proba[pred_i]), 4),
        "explanation": expl,
        "model": bundle.name,
    }


def hormonal_state_to_finding(pred: dict, mode: AnalysisMode = AnalysisMode.causal) -> Finding:
    top = ", ".join(
        f"{e['feature']}={e['value']}" for e in pred.get("explanation", [])[:3]
    ) or "multi-source features"
    is_pfl = str(pred.get("model") or "").startswith("Personalized_GRU")
    if is_pfl:
        statement = (
            f"A personalized sequence model estimates cycle phase as {pred['predicted_state']} "
            f"(confidence {pred['confidence']:.2f}). Input snapshot: {top}. "
            f"This is an association with wearable/symptom patterns, not a clinical diagnosis "
            f"and not a feature-attribution explanation."
        )
    else:
        statement = (
            f"A multi-source model estimates hormonal state as {pred['predicted_state']} "
            f"(confidence {pred['confidence']:.2f}). Top contributing signals: {top}. "
            f"This is an association with wearable/symptom patterns, not a clinical diagnosis."
        )
    strength = (
        Certainty.high if pred["confidence"] >= 0.7
        else Certainty.medium if pred["confidence"] >= 0.45
        else Certainty.low
    )
    return Finding(
        finding_id="F_model_hormonal_state",
        finding_type=FindingType.temporal_association,
        title="Model: estimated hormonal state",
        statement=statement,
        establishment=EstablishmentClass.possible,
        strength=strength,
        analysis_mode=mode,
        method=f"cyclebench.model:{pred['model']}",
        supporting_event_ids=["model:hormonal_state"],
        source_ids=["model:hormonal_state_v0.1"],
        metrics={
            "predicted_state": pred["predicted_state"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "explanation": pred["explanation"],
            "evidence_class": EvidenceClass.inferred.value,
        },
        limitation=(
            "Trained on mcPHASES (n=42, young adults). Not validated as a diagnostic device. "
            "When model=Personalized_GRU_FedPer, short history may use pad×5; see docs/PFL.md."
        ),
    )


def menopause_stage_to_finding(pred: dict, mode: AnalysisMode = AnalysisMode.causal) -> Finding:
    top = ", ".join(
        f"{e['feature']}={e['value']}" for e in pred.get("explanation", [])[:3]
    ) or "hormone and symptom features"
    pretty = pred["predicted_stage"].replace("_", " ")
    statement = (
        f"A menopause-stage model estimates category '{pretty}' "
        f"(confidence {pred['confidence']:.2f}). Top contributing signals: {top}. "
        f"This estimates a stage category from hormones/symptoms/age; it does not diagnose "
        f"menopause or predict exact onset date."
    )
    strength = (
        Certainty.high if pred["confidence"] >= 0.7
        else Certainty.medium if pred["confidence"] >= 0.45
        else Certainty.low
    )
    return Finding(
        finding_id="F_model_menopause_stage",
        finding_type=FindingType.temporal_association,
        title="Model: estimated menopause stage category",
        statement=statement,
        establishment=EstablishmentClass.possible,
        strength=strength,
        analysis_mode=mode,
        method=f"cyclebench.model:{pred['model']}",
        supporting_event_ids=["model:menopause_stage"],
        source_ids=["model:menopause_stage_v0.1"],
        metrics={
            "predicted_stage": pred["predicted_stage"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "explanation": pred["explanation"],
            "evidence_class": EvidenceClass.inferred.value,
        },
        limitation=(
            "Stage category estimate only. Re-train on real SWAN ICPSR export for "
            "publication-grade metrics; synthetic fallback is for offline demo/CI."
        ),
    )
