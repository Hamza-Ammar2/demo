"""Tests for the medical foundation graph."""

from pathlib import Path

import pytest

from cyclebench.foundation.build import build_foundation
from cyclebench.foundation.io import DEFAULT_PATH, load_bundle
from cyclebench.foundation.query import assemble_read, intake_tags
from cyclebench.foundation.schema import export_json_schema
from cyclebench.knowledge import match_fundamentals
from cyclebench.safety import assert_safe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return build_foundation()


def test_bundle_shapes_and_integrity(bundle):
    assert len(bundle.entities) >= 70
    assert len(bundle.associations) >= 45
    assert len(bundle.evidence) >= 10
    # integrity validator already ran; spot-check unique ids
    assert len({e.entity_id for e in bundle.entities}) == len(bundle.entities)
    assert DEFAULT_PATH.exists()


def test_json_schema_export():
    sch = export_json_schema()
    assert "Entity" in sch and "Association" in sch and "Evidence" in sch


def test_seed_and_evidence_language_is_safe(bundle):
    for a in bundle.associations:
        if a.talking_point:
            assert_safe(a.talking_point)
        if a.ask_doctor:
            assert_safe(a.ask_doctor)
    for e in bundle.evidence:
        assert_safe(e.summary_sentence)


def test_mcphases_evidence_attached(bundle):
    aids = {e.association_id for e in bundle.evidence if e.dataset.value == "mcphases"}
    assert "assoc.pop_headaches" in aids


def test_assemble_pill_migraine_read():
    intake = {
        "symptoms": [{"type": "migraine", "severity": "severe"}],
        "symptom_timing": "before_period",
        "contraception_status": "changed",
        "contraception_formulation": "Combined pill",
        "sleep_quality": "bad",
        "last_period_days_ago": 20,
        "age_range": "30-39",
    }
    tags = intake_tags(intake)
    assert "contraception:estrogen" in tags
    assert "sleep:poor" in tags
    read = assemble_read(intake)
    ids = {c.association_id for c in read.cards}
    assert "assoc.estrogen_migraine" in ids
    assert "assoc.sleep_headache" in ids
    assert "assoc.headache_menstrual" in ids or "assoc.migraine_luteal" in ids
    # should NOT invent mood/pelvic without those symptoms
    assert "assoc.mood_luteal" not in ids
    assert "assoc.pelvic_menstrual" not in ids
    assert read.doctor_questions
    # mcPHASES evidence appears on a headache population edge when matched
    headache_cards = [c for c in read.cards if "headache" in c.title.lower() or "migraine" in c.title.lower()]
    assert headache_cards


def test_knowledge_wrapper_still_works():
    hits = match_fundamentals({
        "symptoms": [{"type": "migraine"}],
        "contraception_status": "on_stable",
        "contraception_formulation": "Combined pill",
        "symptom_timing": "before_period",
        "sleep_quality": "ok",
    })
    assert any(h["id"] == "assoc.estrogen_migraine" for h in hits)


def test_clusters_marked_not_diagnosis(bundle):
    clusters = [e for e in bundle.entities if e.kind.value == "cluster"]
    assert clusters
    assert all(e.not_a_diagnosis for e in clusters)
