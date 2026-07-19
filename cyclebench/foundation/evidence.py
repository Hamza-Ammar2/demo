"""Attach dataset evidence onto foundation associations.

Adapters never invent new clinical associations — they only strengthen existing edges
with numbers, ranges, model hooks, or optional Q&A citations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cyclebench.foundation.schema import (
    AssociationStatus,
    DatasetTag,
    Evidence,
    EvidenceType,
    FoundationBundle,
)
from cyclebench.safety import assert_safe

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "results" / "reference_stats.json"
NHANES_RANGES = ROOT / "data" / "nhanes_harmonized" / "reference_ranges.csv"
MEDQUAD_WH = ROOT / "data" / "qa" / "womens_health_qa.jsonl"

# mcPHASES column -> association_id for population_enriched_in edges
MCPHASES_ASSOC = {
    "headaches": "assoc.pop_headaches",
    "cramps": "assoc.pop_cramps",
    "fatigue": "assoc.pop_fatigue",
    "moodswing": "assoc.pop_moodswing",
    "bloating": "assoc.pop_bloating",
    "sorebreasts": "assoc.pop_sorebreasts",
}

NHANES_ANALYTE_ENTITY = {
    "estradiol": "horm.estradiol",
    "fsh": "horm.fsh",
    "lh": "horm.lh",
    "shbg": "horm.shbg",
    "progesterone": "horm.progesterone",
}


def attach_mcphases_evidence(bundle: FoundationBundle) -> list[Evidence]:
    if not REF.exists():
        return []
    ref = json.loads(REF.read_text())
    src = ref.get("sources", {}).get("mcphases", {})
    n = src.get("n_participants")
    out: list[Evidence] = []
    for col, aid in MCPHASES_ASSOC.items():
        if aid not in bundle.association_map():
            continue
        stat = ref.get("symptom_phase", {}).get(col)
        if not stat:
            continue
        phase = stat.get("dominant_phase")
        p = stat.get("p_value")
        sig = bool(stat.get("significant"))
        if sig:
            sentence = (
                f"In the mcPHASES cohort (n={n}), reported {col} clustered most in the "
                f"{str(phase).lower()} phase (χ²={stat.get('chi2')}, p={p})."
            )
        else:
            sentence = (
                f"In the mcPHASES cohort (n={n}), reported {col} did not show a clear "
                f"phase cluster (p={p})."
            )
        assert_safe(sentence, where=f"mcphases:{col}")
        out.append(Evidence(
            evidence_id=f"ev.mcphases.{col}",
            association_id=aid,
            evidence_type=EvidenceType.cohort_rate,
            dataset=DatasetTag.mcphases,
            metrics={
                "n_participants": n,
                "n_episodes": stat.get("n_episodes"),
                "episode_rate_by_phase": stat.get("episode_rate_by_phase"),
                "dominant_phase": phase,
                "chi2": stat.get("chi2"),
                "p_value": p,
                "significant": sig,
            },
            summary_sentence=sentence,
            provenance="results/reference_stats.json from local mcPHASES aggregates",
            license_note="PhysioNet restricted — aggregate statistics only; no participant rows.",
        ))
        # mark association evidenced
        for a in bundle.associations:
            if a.association_id == aid:
                a.status = AssociationStatus.evidenced
    return out


def attach_nhanes_evidence(bundle: FoundationBundle) -> list[Evidence]:
    if not NHANES_RANGES.exists():
        return []
    import pandas as pd
    df = pd.read_csv(NHANES_RANGES)
    # Attach ranges as evidence on marker associations where we have a clear edge,
    # and also create guideline_seed-style summaries on hormone marker associations.
    target_assoc = {
        "fsh": "assoc.fsh_meno",
        "estradiol": "assoc.e2_meno",
    }
    out: list[Evidence] = []
    for analyte, aid in target_assoc.items():
        if aid not in bundle.association_map():
            continue
        sub = df[df["analyte"] == analyte]
        if sub.empty:
            continue
        # pick a midlife band if present
        band = sub[sub["age_band"].isin(["48-57", "38-47", "40-49"])]
        row = band.iloc[0] if len(band) else sub.iloc[0]
        sentence = (
            f"NHANES female reference (age band {row['age_band']}, n={int(row['n'])}): "
            f"{analyte} median {row['median']} {row['unit']} "
            f"(2.5th–97.5th percentile {row['p2_5']}–{row['p97_5']}). "
            f"Cross-sectional population ranges — not personal targets."
        )
        assert_safe(sentence, where=f"nhanes:{analyte}")
        out.append(Evidence(
            evidence_id=f"ev.nhanes.range.{analyte}",
            association_id=aid,
            evidence_type=EvidenceType.reference_range,
            dataset=DatasetTag.nhanes,
            metrics=row.to_dict(),
            summary_sentence=sentence,
            provenance="data/nhanes_harmonized/reference_ranges.csv",
            license_note="NHANES public domain; harmonized export CC-BY-4.0.",
        ))
        for a in bundle.associations:
            if a.association_id == aid:
                a.status = AssociationStatus.evidenced
    return out


def attach_model_hooks(bundle: FoundationBundle) -> list[Evidence]:
    """Static model-capability evidence (runtime predictions added in assemble)."""
    out: list[Evidence] = []
    pcos_path = ROOT / "models" / "pcos_risk_v0.1.joblib"
    if "assoc.model_pcos" in bundle.association_map() and pcos_path.exists():
        metrics_path = ROOT / "results" / "model_pcos_risk.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        sentence = (
            "A CycleBench PCOS-risk model is available (trained on 541 Kerala clinic patients). "
            f"Held-out balanced accuracy ≈ {metrics.get('balanced_accuracy', 'n/a')}. "
            "Research screening signal only — not a diagnosis."
        )
        assert_safe(sentence, where="model:pcos")
        out.append(Evidence(
            evidence_id="ev.model.pcos_risk.capability",
            association_id="assoc.model_pcos",
            evidence_type=EvidenceType.model_signal,
            dataset=DatasetTag.pcos_kaggle,
            metrics={"balanced_accuracy": metrics.get("balanced_accuracy"),
                     "top_features": metrics.get("feature_importances", [])[:5]},
            summary_sentence=sentence,
            provenance="models/pcos_risk_v0.1.joblib + results/model_pcos_risk.json",
            license_note="Model checkpoint derived from Kaggle PCOS dataset (not redistributed).",
        ))
    meno_path = ROOT / "models" / "menopause_stage_v0.1.joblib"
    if "assoc.model_meno" in bundle.association_map() and meno_path.exists():
        metrics_path = ROOT / "results" / "model_menopause_stage.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        src = metrics.get("data_source", "unknown")
        sentence = (
            "A CycleBench menopause-stage model is available for midlife category estimates. "
            f"Training data source: {src}. Stage category estimate only — not a diagnosis."
        )
        assert_safe(sentence, where="model:meno")
        out.append(Evidence(
            evidence_id="ev.model.meno_stage.capability",
            association_id="assoc.model_meno",
            evidence_type=EvidenceType.model_signal,
            dataset=DatasetTag.swan if "swan" in str(src).lower() else DatasetTag.other,
            metrics={"balanced_accuracy": metrics.get("balanced_accuracy"), "data_source": src},
            summary_sentence=sentence,
            provenance="models/menopause_stage_v0.1.joblib",
            license_note="If synthetic_swan_like, metrics are illustrative until real SWAN is loaded.",
        ))
    return out


def attach_medquad_citations(bundle: FoundationBundle, limit_per_assoc: int = 1) -> list[Evidence]:
    """Optional phrasing citations — never create associations."""
    if not MEDQUAD_WH.exists():
        return []
    # Map simple keywords to association ids
    keyword_map = {
        "migraine": "assoc.estrogen_migraine",
        "menopause": "assoc.midlife_composite",
        "polycystic": "assoc.irregular_pcos",
        "thyroid": "assoc.tsh_fatigue",
        "amenorrhea": "assoc.amenorrhea_eval",
    }
    buckets: dict[str, list[dict]] = {k: [] for k in keyword_map}
    with MEDQUAD_WH.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = (rec.get("question", "") + " " + rec.get("focus", "")).lower()
            for kw in keyword_map:
                if kw in blob and len(buckets[kw]) < limit_per_assoc:
                    buckets[kw].append(rec)
    out: list[Evidence] = []
    for kw, aid in keyword_map.items():
        if aid not in bundle.association_map():
            continue
        for i, rec in enumerate(buckets[kw]):
            q = (rec.get("question") or "")[:180]
            sentence = f"Related NIH-sourced Q&A topic on file: “{q}” (citation for phrasing context only)."
            # may contain edge words — soft fail
            try:
                assert_safe(sentence, where=f"medquad:{kw}")
            except Exception:
                continue
            out.append(Evidence(
                evidence_id=f"ev.medquad.{kw}.{i}",
                association_id=aid,
                evidence_type=EvidenceType.qa_citation,
                dataset=DatasetTag.medquad,
                metrics={"focus": rec.get("focus"), "source": rec.get("source")},
                summary_sentence=sentence,
                provenance="data/qa/womens_health_qa.jsonl (MedQuAD subset)",
                license_note="MedQuAD from NIH website-derived QA; citation only.",
            ))
    return out


def attach_all_evidence(bundle: FoundationBundle) -> FoundationBundle:
    evidence = []
    evidence += attach_mcphases_evidence(bundle)
    evidence += attach_nhanes_evidence(bundle)
    evidence += attach_model_hooks(bundle)
    evidence += attach_medquad_citations(bundle)
    # guideline_seed evidence for high-priority counseling edges
    for aid in ("assoc.estrogen_migraine", "assoc.amenorrhea_eval", "assoc.midlife_composite"):
        a = bundle.association_map().get(aid)
        if not a:
            continue
        sentence = f"Guideline/textbook seed: {a.source}"
        assert_safe(sentence, where=f"guideline:{aid}")
        evidence.append(Evidence(
            evidence_id=f"ev.guideline.{aid}",
            association_id=aid,
            evidence_type=EvidenceType.guideline_seed,
            dataset=DatasetTag.guideline,
            metrics={},
            summary_sentence=sentence,
            provenance=a.source,
            license_note="Curated seed citation — not a full guideline reprint.",
        ))
    bundle.evidence = evidence
    return bundle
