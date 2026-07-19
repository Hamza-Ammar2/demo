"""Shared training utilities: participant-safe splits, metrics, explainability, persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


@dataclass
class ModelBundle:
    name: str
    task: str
    pipeline: Any
    label_encoder: LabelEncoder
    feature_names: list[str]
    classes: list[str]
    metrics: dict
    notes: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "task": self.task,
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "classes": self.classes,
                "metrics": self.metrics,
                "notes": self.notes,
            },
            path,
        )
        meta = {
            "name": self.name,
            "task": self.task,
            "classes": self.classes,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "notes": self.notes,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_bundle(path: Path) -> ModelBundle:
    obj = joblib.load(path)
    return ModelBundle(**obj)


def participant_split(
    participant_ids: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole participants (no person leakage across train/test)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(participant_ids)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_size)))
    test_ids = set(uniq[:n_test])
    test_mask = np.array([p in test_ids for p in participant_ids])
    return ~test_mask, test_mask


def make_classifier(kind: str = "gbm", seed: int = 42) -> Pipeline:
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    else:
        clf = GradientBoostingClassifier(
            random_state=seed, max_depth=3, learning_rate=0.08, n_estimators=120,
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def evaluate_classifier(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels,
        "report": classification_report(
            y_true, y_pred, target_names=labels, zero_division=0, output_dict=True
        ),
    }


def top_importances(pipeline: Pipeline, feature_names: list[str], k: int = 12) -> list[dict]:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    else:
        return []
    order = np.argsort(imp)[::-1][:k]
    return [
        {"feature": feature_names[i], "importance": round(float(imp[i]), 5)}
        for i in order if imp[i] > 0
    ]


def explain_instance(
    pipeline: Pipeline,
    feature_names: list[str],
    x_row: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Cheap local explanation: global importances × |standardized feature| magnitude."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return []
    # transform through imputer+scaler
    x2 = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(x_row.reshape(1, -1))
    )[0]
    score = np.abs(x2) * np.asarray(clf.feature_importances_, dtype=float)
    order = np.argsort(score)[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "contribution": round(float(score[i]), 5),
            "value": None if np.isnan(x_row[i]) else round(float(x_row[i]), 4),
        }
        for i in order if score[i] > 0
    ]
