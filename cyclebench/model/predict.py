"""Inference + schema Finding wrappers for Layer 02 models."""

from __future__ import annotations

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


def predict_hormonal_state(
    features: dict[str, float],
    bundle_path: Optional[Path] = None,
) -> dict:
    path = bundle_path or _default_path("hormonal_state_v0.1")
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint {path}; run `make train-models`")
    bundle = load_bundle(path)
    x = np.array([features.get(f, np.nan) for f in bundle.feature_names], dtype=float)
    proba = bundle.pipeline.predict_proba(x.reshape(1, -1))[0]
    pred_i = int(np.argmax(proba))
    label = str(bundle.classes[pred_i])
    expl = explain_instance(bundle.pipeline, bundle.feature_names, x, top_k=5)
    
    # Collect patient symptom data into local logs
    try:
        from cyclebench.model.pfl import save_local_log
        save_local_log(features, label)
    except Exception:
        pass
        
    # Check if local sequential pFL model is trained and has history
    try:
        from cyclebench.model.pfl import run_pfl_inference
        pfl_pred = run_pfl_inference(features)
        return pfl_pred
    except Exception:
        pass

    return {
        "task": bundle.task,
        "predicted_state": label,
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(bundle.classes, proba)},
        "confidence": round(float(proba[pred_i]), 4),
        "explanation": expl,
        "model": bundle.name,
    }


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
            "Trained on mcPHASES (n=42, young adults). Not validated as a diagnostic device."
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
