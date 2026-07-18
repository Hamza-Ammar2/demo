"""Train Layer 02 models and write checkpoints + metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from cyclebench.model.common import (
    MODELS_DIR,
    RESULTS_DIR,
    ModelBundle,
    evaluate_classifier,
    make_classifier,
    participant_split,
    top_importances,
)
from cyclebench.model.features_mcphases import build_mcphases_table, matrix_from_table as mcp_matrix
from cyclebench.model.features_swan import build_menopause_table, matrix_from_table as swan_matrix


def train_hormonal_state(seed: int = 42) -> ModelBundle:
    df = build_mcphases_table()
    X, y, pid, feats = mcp_matrix(df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    train_m, test_m = participant_split(pid, test_size=0.25, seed=seed)
    pipe = make_classifier("gbm", seed=seed)
    pipe.fit(X[train_m], y_enc[train_m])
    pred = pipe.predict(X[test_m])
    labels = list(le.classes_)
    metrics = evaluate_classifier(y_enc[test_m], pred, labels)
    metrics["n_train_rows"] = int(train_m.sum())
    metrics["n_test_rows"] = int(test_m.sum())
    metrics["n_train_participants"] = int(len(set(pid[train_m])))
    metrics["n_test_participants"] = int(len(set(pid[test_m])))
    metrics["feature_importances"] = top_importances(pipe, feats)
    metrics["split"] = "participant_held_out_25pct"
    metrics["chance_balanced_accuracy"] = round(1.0 / max(len(labels), 1), 4)
    metrics["leakage_guard"] = (
        "Mira hormone metabolites (lh/estrogen/pdg) excluded from features; "
        "participant-level split (no person in both train and test)."
    )
    maj = Counter(y_enc[train_m]).most_common(1)[0][0]
    maj_pred = np.full(int(test_m.sum()), maj)
    metrics["majority_baseline_accuracy"] = round(
        float((maj_pred == y_enc[test_m]).mean()), 4
    )
    bundle = ModelBundle(
        name="hormonal_state_v0.1",
        task="hormonal_state_phase",
        pipeline=pipe,
        label_encoder=le,
        feature_names=feats,
        classes=labels,
        metrics=metrics,
        notes=(
            "Multi-source explainable classifier: wearables + sleep + symptoms + CGM "
            "→ Mira cycle phase. Association-only; not a diagnosis."
        ),
    )
    out = MODELS_DIR / "hormonal_state_v0.1.joblib"
    bundle.save(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_hormonal_state.json").write_text(json.dumps(metrics, indent=2))
    return bundle


def train_menopause_stage(seed: int = 42, allow_synthetic: bool = True) -> ModelBundle:
    df = build_menopause_table(allow_synthetic=allow_synthetic)
    X, y, pid, feats = swan_matrix(df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    train_m, test_m = participant_split(pid, test_size=0.25, seed=seed)
    pipe = make_classifier("gbm", seed=seed)
    pipe.fit(X[train_m], y_enc[train_m])
    pred = pipe.predict(X[test_m])
    labels = list(le.classes_)
    metrics = evaluate_classifier(y_enc[test_m], pred, labels)
    metrics["n_train_rows"] = int(train_m.sum())
    metrics["n_test_rows"] = int(test_m.sum())
    metrics["n_train_participants"] = int(len(set(pid[train_m])))
    metrics["n_test_participants"] = int(len(set(pid[test_m])))
    metrics["feature_importances"] = top_importances(pipe, feats)
    metrics["data_source"] = str(df["data_source"].iloc[0]) if "data_source" in df.columns else "unknown"
    metrics["split"] = "participant_held_out_25pct"
    metrics["disclaimer"] = (
        "Estimates menopausal *stage category* from hormones + symptoms + age. "
        "Not a clinical diagnosis of menopause onset timing."
    )
    bundle = ModelBundle(
        name="menopause_stage_v0.1",
        task="menopause_stage",
        pipeline=pipe,
        label_encoder=le,
        feature_names=feats,
        classes=labels,
        metrics=metrics,
        notes=(
            "Menopause-stage classifier (SWAN public-use when present; synthetic "
            "SWAN-like cohort otherwise). Re-train on real ICPSR export for publication metrics."
        ),
    )
    out = MODELS_DIR / "menopause_stage_v0.1.joblib"
    bundle.save(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_menopause_stage.json").write_text(json.dumps(metrics, indent=2))
    return bundle


def train_all() -> dict:
    out = {}
    try:
        hs = train_hormonal_state()
        out["hormonal_state"] = {
            "ok": True,
            "balanced_accuracy": hs.metrics["balanced_accuracy"],
            "macro_f1": hs.metrics["macro_f1"],
            "path": "models/hormonal_state_v0.1.joblib",
        }
    except FileNotFoundError as e:
        out["hormonal_state"] = {"ok": False, "error": str(e)}

    ms = train_menopause_stage(allow_synthetic=True)
    out["menopause_stage"] = {
        "ok": True,
        "balanced_accuracy": ms.metrics["balanced_accuracy"],
        "macro_f1": ms.metrics["macro_f1"],
        "data_source": ms.metrics.get("data_source"),
        "path": "models/menopause_stage_v0.1.joblib",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_train_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    summary = train_all()
    print(json.dumps(summary, indent=2))
    return 0 if all(v.get("ok") for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
