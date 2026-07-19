"""Medical foundation graph schema — the substrate datasets plug into.

Entities + Associations (seeded from guidelines/textbook associations) + Evidence
(attached from female-specialized datasets). The LLM never creates edges.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class EntityKind(str, Enum):
    symptom = "symptom"
    hormone = "hormone"
    analyte = "analyte"
    cycle_phase = "cycle_phase"
    life_stage = "life_stage"
    intervention = "intervention"
    confounder = "confounder"
    cluster = "cluster"
    other = "other"


class Relation(str, Enum):
    temporally_associated_with = "temporally_associated_with"
    modulated_by = "modulated_by"
    counseling_relevant_with = "counseling_relevant_with"
    marker_of = "marker_of"
    confounded_by = "confounded_by"
    population_enriched_in = "population_enriched_in"
    predicted_by_model = "predicted_by_model"


class StrengthPrior(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AssociationStatus(str, Enum):
    seed = "seed"
    evidenced = "evidenced"
    model_linked = "model_linked"


class EvidenceType(str, Enum):
    cohort_rate = "cohort_rate"
    reference_range = "reference_range"
    model_signal = "model_signal"
    guideline_seed = "guideline_seed"
    qa_citation = "qa_citation"


class DatasetTag(str, Enum):
    mcphases = "mcphases"
    nhanes = "nhanes"
    pcos_kaggle = "pcos_kaggle"
    swan = "swan"
    medquad = "medquad"
    guideline = "guideline"
    other = "other"


class Entity(BaseModel):
    entity_id: str
    kind: EntityKind
    label: str
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""
    notes: str = ""
    not_a_diagnosis: bool = False  # True for discussion clusters


class Association(BaseModel):
    association_id: str
    subject_id: str
    object_id: str
    relation: Relation
    directionality: str = "bidirectional"  # subject->object | object->subject | bidirectional
    strength_prior: StrengthPrior = StrengthPrior.medium
    population_scope: str = "general"
    caveats: str = ""
    source: str
    status: AssociationStatus = AssociationStatus.seed
    # Intake matching + doctor questions (first-class)
    match_tags: list[str] = Field(default_factory=list)
    talking_point: str = ""
    ask_doctor: str = ""
    title: str = ""


class Evidence(BaseModel):
    evidence_id: str
    association_id: str
    evidence_type: EvidenceType
    dataset: DatasetTag
    metrics: dict[str, Any] = Field(default_factory=dict)
    summary_sentence: str
    provenance: str = ""
    license_note: str = ""


class FoundationBundle(BaseModel):
    version: str = "v0.1"
    description: str = (
        "CycleBench medical foundation: guideline-seeded associations strengthened "
        "by female-specialized dataset evidence. Not a diagnostic system."
    )
    entities: list[Entity] = Field(default_factory=list)
    associations: list[Association] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _integrity(self) -> "FoundationBundle":
        ids = {e.entity_id for e in self.entities}
        if len(ids) != len(self.entities):
            raise ValueError("duplicate entity_id in foundation")
        aids = {a.association_id for a in self.associations}
        if len(aids) != len(self.associations):
            raise ValueError("duplicate association_id in foundation")
        for a in self.associations:
            if a.subject_id not in ids or a.object_id not in ids:
                raise ValueError(
                    f"dangling association {a.association_id}: "
                    f"{a.subject_id} -> {a.object_id}"
                )
        for ev in self.evidence:
            if ev.association_id not in aids:
                raise ValueError(
                    f"evidence {ev.evidence_id} points to missing association "
                    f"{ev.association_id}"
                )
        return self

    def entity_map(self) -> dict[str, Entity]:
        return {e.entity_id: e for e in self.entities}

    def association_map(self) -> dict[str, Association]:
        return {a.association_id: a for a in self.associations}

    def evidence_for(self, association_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.association_id == association_id]


class FoundationCard(BaseModel):
    """One assembled UI card: foundation + evidence + optional personal pattern."""
    association_id: str
    title: str
    foundation_fact: str
    evidence_summaries: list[str] = Field(default_factory=list)
    personal_pattern: Optional[str] = None
    ask_doctor: str = ""
    source: str = ""
    datasets: list[str] = Field(default_factory=list)
    strength_prior: str = "medium"
    relation: str = ""


class FoundationRead(BaseModel):
    matched_entity_ids: list[str] = Field(default_factory=list)
    cards: list[FoundationCard] = Field(default_factory=list)
    doctor_questions: list[str] = Field(default_factory=list)
    model_signals: list[dict[str, Any]] = Field(default_factory=list)
    missing_prompts: list[str] = Field(default_factory=list)
    foundation_version: str = "v0.1"


def export_json_schema() -> dict:
    return {
        "Entity": Entity.model_json_schema(),
        "Association": Association.model_json_schema(),
        "Evidence": Evidence.model_json_schema(),
        "FoundationBundle": FoundationBundle.model_json_schema(),
        "FoundationRead": FoundationRead.model_json_schema(),
    }
