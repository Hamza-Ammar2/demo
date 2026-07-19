"""The model factory: feed it a dataset spec, get a reproducible, explainable model.

This is the reusable Layer-02 foundation. Adding a new dataset is *declarative* — you
describe a `TabularTask` (where the data is, the target, the population, what it does and
does not support) and the harness handles everything else:

  load -> clean -> leakage-safe split -> train -> evaluate -> explain -> persist
       -> schema-typed Finding (with provenance + population + limitations)

Key honesty principle (matches docs/MEDICAL_SAFETY.md): a model only knows the dataset
it was trained on and the population it came from. Every task declares that explicitly,
and every prediction carries it downstream. Models produce *associations*, never diagnoses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cyclebench.model.common import (
    MODELS_DIR,
    RESULTS_DIR,
    ModelBundle,
    evaluate_classifier,
    explain_instance,
    load_bundle,
    make_classifier,
    participant_split,
    top_importances,
)
from cyclebench.schema import (
    AnalysisMode,
    Certainty,
    EstablishmentClass,
    EvidenceClass,
    Finding,
    FindingType,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TabularTask:
    """Declarative spec for one supervised tabular model."""

    name: str
    version: str
    data_path: Path
    target: str
    population: str
    provenance: str
    outcome_phrase: str = "the target outcome"      # human phrasing for the positive/predicted class
    positive_label: Optional[object] = None          # for binary tasks: which class is "positive"
    id_column: Optional[str] = None                  # group-safe split key (else stratified rows)
    feature_columns: Optional[list[str]] = None      # None => auto (all numeric except target/ids/drops)
    drop_columns: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    limitations: str = ""
    classifier: str = "gbm"
    clean: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    @property
    def bundle_path(self) -> Path:
        return MODELS_DIR / f"{self.name}_{self.version}.joblib"


# --------------------------------------------------------------------------- #
# data loading + matrix building
# --------------------------------------------------------------------------- #
def load_frame(task: TabularTask) -> pd.DataFrame:
    df = pd.read_csv(task.data_path)
    df.columns = [" ".join(str(c).split()).strip() for c in df.columns]  # normalize whitespace
    if task.clean:
        df = task.clean(df)
    return df


def build_matrix(task: TabularTask, df: pd.DataFrame):
    if task.target not in df.columns:
        raise KeyError(f"target '{task.target}' not in columns")
    y_raw = df[task.target]

    exclude = {task.target, task.id_column, *task.drop_columns}
    exclude |= {c for c in df.columns if c.lower().startswith("unnamed")}

    if task.feature_columns:
        feats = [c for c in task.feature_columns if c in df.columns]
    else:
        feats = [c for c in df.columns if c not in exclude]

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    # drop columns that are entirely missing after coercion
    good = [c for c in feats if not X[c].isna().all()]
    X = X[good].to_numpy(dtype=float)

    mask = y_raw.notna().to_numpy()
    y = y_raw[mask].astype(str).to_numpy()
    X = X[mask]
    ids = df[task.id_column][mask].to_numpy() if task.id_column else None
    return X, y, ids, good


# --------------------------------------------------------------------------- #
# train / evaluate / persist
# --------------------------------------------------------------------------- #
def train_task(task: TabularTask, seed: int = 42) -> ModelBundle:
    from collections import Counter

    df = load_frame(task)
    X, y, ids, feats = build_matrix(task, df)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels = list(le.classes_)

    if ids is not None:
        train_m, test_m = participant_split(ids, test_size=0.25, seed=seed)
        train_idx = np.where(train_m)[0]
        test_idx = np.where(test_m)[0]
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y_enc)), test_size=0.25, random_state=seed,
            stratify=y_enc if len(set(y_enc)) > 1 else None,
        )

    # held-out evaluation
    clf = make_classifier(task.classifier, seed=seed)
    clf.fit(X[train_idx], y_enc[train_idx])
    y_pred = clf.predict(X[test_idx])
    metrics = evaluate_classifier(y_enc[test_idx], y_pred, labels)

    # honest baselines for context
    metrics["chance_balanced_accuracy"] = round(1.0 / max(len(labels), 1), 4)
    maj = Counter(y_enc[train_idx]).most_common(1)[0][0]
    metrics["majority_baseline_accuracy"] = round(
        float((np.full(len(test_idx), maj) == y_enc[test_idx]).mean()), 4
    )
    metrics["n_train_rows"] = int(len(train_idx))
    metrics["n_test_rows"] = int(len(test_idx))
    metrics["split"] = "participant_held_out_25pct" if ids is not None else "stratified_row_25pct"
    metrics["population"] = task.population
    metrics["provenance"] = task.provenance
    metrics["feature_importances"] = top_importances(clf, feats, k=15)
    if task.name == "pcos_risk":
        metrics["data_source"] = "kaggle_pcos"
        metrics["data_redistributable"] = False
        metrics["disclaimer"] = (
            "Trained on the Kaggle PCOS clinic cohort. Raw CSV is not redistributed; "
            "re-download to retrain. Research/demo risk signal only — not a clinical diagnosis."
        )

    # deploy model: refit on all rows (metrics above are the held-out estimate)
    final = make_classifier(task.classifier, seed=seed)
    final.fit(X, y_enc)

    bundle = ModelBundle(
        name=f"{task.name}_{task.version}",
        task=task.name,
        pipeline=final,
        label_encoder=le,
        feature_names=feats,
        classes=[str(c) for c in labels],
        metrics=metrics,
        notes=f"{task.provenance}. Population: {task.population}.",
    )
    bundle.save(task.bundle_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"model_{task.name}.json").write_text(json.dumps(metrics, indent=2))
    return bundle


# --------------------------------------------------------------------------- #
# inference + schema Finding
# --------------------------------------------------------------------------- #
def predict_task(task_name: str, features: dict) -> dict:
    task = TASKS[task_name]
    if not task.bundle_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {task.bundle_path}; run `make train-tasks`")
    bundle = load_bundle(task.bundle_path)
    x = np.array([features.get(f, np.nan) for f in bundle.feature_names], dtype=float)
    proba = bundle.pipeline.predict_proba(x.reshape(1, -1))[0]
    pred_i = int(np.argmax(proba))
    probs = {str(c): round(float(p), 4) for c, p in zip(bundle.classes, proba)}

    pos_prob = None
    if task.positive_label is not None:
        pos = str(task.positive_label)
        pos_prob = probs.get(pos)

    return {
        "task": task.name,
        "predicted_label": str(bundle.classes[pred_i]),
        "positive_probability": pos_prob,
        "confidence": round(float(proba[pred_i]), 4),
        "probabilities": probs,
        "explanation": explain_instance(bundle.pipeline, bundle.feature_names, x, top_k=6),
        "model": bundle.name,
        "population": task.population,
    }


def task_to_finding(task_name: str, pred: dict, mode: AnalysisMode = AnalysisMode.causal) -> Finding:
    task = TASKS[task_name]
    top = ", ".join(e["feature"] for e in pred.get("explanation", [])[:3]) or "input features"
    if pred.get("positive_probability") is not None:
        p = pred["positive_probability"]
        outcome = f"a pattern consistent with {task.outcome_phrase} (probability {p:.2f})"
        conf = p
    else:
        outcome = f"{task.outcome_phrase}: {pred['predicted_label']} (confidence {pred['confidence']:.2f})"
        conf = pred["confidence"]

    statement = (
        f"A {task.name} model estimates {outcome}. Top contributing signals: {top}. "
        f"This is a statistical association learned from {task.population}; "
        f"it is not a diagnosis."
    )
    strength = (
        Certainty.high if conf >= 0.7 else Certainty.medium if conf >= 0.45 else Certainty.low
    )
    return Finding(
        finding_id=f"F_model_{task.name}",
        finding_type=FindingType.temporal_association,
        title=f"Model: {task.name}",
        statement=statement,
        establishment=EstablishmentClass.possible,
        strength=strength,
        analysis_mode=mode,
        method=f"cyclebench.model.tasks:{pred['model']}",
        supporting_event_ids=[f"model:{task.name}"],
        source_ids=[f"model:{task.name}_{task.version}"],
        metrics={
            "predicted_label": pred["predicted_label"],
            "positive_probability": pred.get("positive_probability"),
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "explanation": pred["explanation"],
            "evidence_class": EvidenceClass.inferred.value,
        },
        limitation=f"{task.limitations} Trained only on: {task.population}. "
                   f"Applies to inputs resembling that population; not validated as a device.",
    )


# --------------------------------------------------------------------------- #
# task registry — feeding a new dataset = adding an entry here
# --------------------------------------------------------------------------- #
def _clean_pcos(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")


PCOS = TabularTask(
    name="pcos_risk",
    version="v0.1",
    data_path=ROOT / "data" / "kaggle" / "pcos-dataset" / "PCOS_data.csv",
    target="PCOS (Y/N)",
    positive_label="1",
    outcome_phrase="PCOS",
    population="541 patients from clinics in Kerala, India (Kaggle PCOS dataset)",
    provenance="Kaggle: shreyasvedpathak/pcos-dataset",
    id_column=None,
    drop_columns=["Sl. No", "Patient File No."],
    supports=[
        "PCOS-risk association from clinical + hormonal + symptom features",
        "explainable screening signal for research/education",
    ],
    does_not_support=[
        "clinical diagnosis of PCOS",
        "use outside a similar clinical population",
        "menopause or cycle-phase prediction",
    ],
    limitations="Single-clinic cross-sectional data; class-imbalanced; not externally validated.",
)

TASKS: dict[str, TabularTask] = {PCOS.name: PCOS}


def train_all_tasks() -> dict:
    out = {}
    for name, task in TASKS.items():
        try:
            b = train_task(task)
            out[name] = {"ok": True, "balanced_accuracy": b.metrics.get("balanced_accuracy")}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": str(e)}
    return out


def main() -> int:
    summary = train_all_tasks()
    print(json.dumps(summary, indent=2))
    return 0 if all(v.get("ok") for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
