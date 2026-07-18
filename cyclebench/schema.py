"""CycleBench data schema (v0.1).

A documented, versioned, reusable schema for longitudinal hormonal-health cases.
Implemented as Pydantic v2 models; JSON Schema is exportable via `export_json_schema()`.

Core entities:
  SubjectProfile  - who the case is about (de-identified)
  SourceReference - where a piece of information came from
  HealthEvent     - a single dated observation/event
  CycleContext    - a menstrual cycle and its (mode-tagged) phase estimate
  Finding         - a deterministic analytical result with full provenance
  DoctorBrief     - the appointment-ready summary assembled from validated findings
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class EvidenceClass(str, Enum):
    """How a value came to be known. Ordered from strongest to weakest evidence."""

    measured = "measured"            # device / assay reading
    documented = "documented"        # clinical record / prescription
    patient_reported = "patient_reported"
    inferred = "inferred"            # derived by the engine, never by an LLM as fact


class Certainty(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    unknown = "unknown"


class SourceType(str, Enum):
    voice = "voice"
    text = "text"
    diary = "diary"
    wearable = "wearable"
    laboratory = "laboratory"
    synthetic_fixture = "synthetic_fixture"


class DatePrecision(str, Enum):
    """Precision of a timestamp, so the timeline compiler can reason about vagueness."""

    exact = "exact"
    day = "day"
    approximate = "approximate"     # "around mid-March"
    range = "range"                 # start/end define a window
    unknown = "unknown"


class EventType(str, Enum):
    symptom = "symptom"
    menstrual_onset = "menstrual_onset"
    estimated_cycle_phase = "estimated_cycle_phase"
    ovulation_proxy = "ovulation_proxy"
    medication_started = "medication_started"
    medication_stopped = "medication_stopped"
    dose_changed = "dose_changed"
    contraception_changed = "contraception_changed"
    sleep_measurement = "sleep_measurement"
    stress_entry = "stress_entry"
    wearable_measurement = "wearable_measurement"
    glucose_measurement = "glucose_measurement"
    hormone_measurement = "hormone_measurement"
    laboratory_result = "laboratory_result"
    clinical_encounter = "clinical_encounter"
    free_text_note = "free_text_note"


class AnalysisMode(str, Enum):
    """Mandatory, no implicit default anywhere it is consumed.

    retrospective: may use later-confirmed information (e.g. next menstrual onset)
                   to reconstruct past phase. Valid for describing past patterns.
    causal:        may use ONLY information available at/before the analysis time.
                   Mandatory for any forward-looking / predictive claim.
    """

    retrospective = "retrospective"
    causal = "causal"


class FindingType(str, Enum):
    temporal_association = "temporal_association"      # pattern repeats in a window
    change_after_event = "change_after_event"          # symptom change after med/contraception
    confounder = "confounder"                          # a competing explanation
    missing_information = "missing_information"         # high-value gap
    data_quality = "data_quality"                      # conflicting/incomplete data


class EstablishmentClass(str, Enum):
    established = "established"
    possible = "possible"
    not_established = "not_established"
    missing = "missing"


class ProvenanceStatus(str, Enum):
    complete = "complete"
    partial = "partial"
    unverified = "unverified"


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
class SubjectProfile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    subject_id: str = Field(..., description="Anonymous, non-identifying subject id.")
    age_range: Optional[str] = Field(
        None, description="Coarse age band, e.g. '30-39'. Never an exact DOB."
    )
    life_stage: Optional[str] = Field(
        None, description="Reproductive/life-stage context when supplied."
    )
    contraception_status: Optional[str] = None
    contraception_formulation: Optional[str] = None
    menopause_status: Optional[str] = None
    hrt_status: Optional[str] = None
    timezone: Optional[str] = None
    source_quality: Certainty = Certainty.unknown


class SourceReference(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source_id: str
    source_type: SourceType
    excerpt: Optional[str] = Field(
        None, description="Original text excerpt or a record pointer (not raw PII)."
    )
    observed_at: Optional[datetime] = None
    confidence: Certainty = Certainty.unknown
    provenance_status: ProvenanceStatus = ProvenanceStatus.unverified


class HealthEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    subject_id: str
    event_type: EventType
    label: Optional[str] = Field(None, description="Human label, e.g. 'severe migraine'.")

    start: Optional[date] = None
    end: Optional[date] = Field(None, description="For ranges/durations.")
    date_precision: DatePrecision = DatePrecision.unknown

    value: Optional[float] = None
    unit: Optional[str] = None
    severity: Optional[float] = Field(
        None, ge=0, le=10, description="0-10 severity where meaningful."
    )
    certainty: Certainty = Certainty.unknown
    evidence_class: EvidenceClass = EvidenceClass.patient_reported

    source_id: Optional[str] = None
    observed_at: Optional[datetime] = Field(
        None, description="When this was recorded/known (drives causal leakage checks)."
    )

    cycle_id: Optional[str] = None
    medication_context: Optional[str] = None

    @model_validator(mode="after")
    def _check_range(self) -> "HealthEvent":
        if self.start and self.end and self.end < self.start:
            raise ValueError(
                f"event {self.event_id}: end {self.end} precedes start {self.start}"
            )
        if self.date_precision == DatePrecision.range and not (self.start and self.end):
            raise ValueError(
                f"event {self.event_id}: range precision requires both start and end"
            )
        return self


class CycleContext(BaseModel):
    schema_version: str = SCHEMA_VERSION
    cycle_id: str
    subject_id: str
    period_onset: Optional[date] = None
    next_known_onset: Optional[date] = Field(
        None, description="Only populated/consumed in retrospective mode."
    )
    estimated_phase: Optional[str] = None
    phase_confidence: Certainty = Certainty.unknown
    analysis_mode: AnalysisMode = Field(
        ..., description="Mandatory. No implicit default."
    )
    anchor_source: Optional[str] = None
    used_future_data: bool = Field(
        False,
        description="True if next_known_onset (future info) informed the phase estimate. "
        "Must be False in causal mode.",
    )

    @model_validator(mode="after")
    def _causal_forbids_future(self) -> "CycleContext":
        if self.analysis_mode == AnalysisMode.causal and self.used_future_data:
            raise ValueError(
                f"cycle {self.cycle_id}: causal mode cannot use future data "
                "(used_future_data=True)"
            )
        if self.analysis_mode == AnalysisMode.causal and self.next_known_onset is not None:
            raise ValueError(
                f"cycle {self.cycle_id}: causal mode cannot reference next_known_onset"
            )
        return self


class Finding(BaseModel):
    schema_version: str = SCHEMA_VERSION
    finding_id: str
    title: str
    statement: str = Field(..., description="Plain-language, association-only phrasing.")
    finding_type: FindingType

    supporting_event_ids: list[str] = Field(default_factory=list)
    contradicting_event_ids: list[str] = Field(default_factory=list)
    confounder_event_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    strength: Certainty = Certainty.unknown
    method: str = Field(..., description="Name of the deterministic method used.")
    analysis_mode: AnalysisMode = Field(..., description="Mandatory.")
    establishment: EstablishmentClass = EstablishmentClass.possible
    limitation: Optional[str] = None

    metrics: dict = Field(
        default_factory=dict,
        description="Transparent numbers behind the finding (episode counts, rates, etc.).",
    )
    provenance_status: ProvenanceStatus = ProvenanceStatus.unverified

    @model_validator(mode="after")
    def _provenance_required(self) -> "Finding":
        # A finding that asserts something (not a 'missing'/'not_established' note)
        # must point at supporting evidence. This is the anti-hallucination gate.
        asserts = self.establishment in (
            EstablishmentClass.established,
            EstablishmentClass.possible,
        )
        if asserts and not self.supporting_event_ids and not self.source_ids:
            raise ValueError(
                f"finding {self.finding_id}: asserting findings require provenance "
                "(supporting_event_ids or source_ids)"
            )
        return self


class DoctorBrief(BaseModel):
    schema_version: str = SCHEMA_VERSION
    subject_id: str
    analysis_mode: AnalysisMode

    opening_statement: str = Field(..., description="~30-second appointment opener.")
    strongest_findings: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    established: list[str] = Field(default_factory=list)
    possible: list[str] = Field(default_factory=list)
    not_established: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    finding_ids: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This prototype organizes health information and detects temporal patterns. "
        "It does not diagnose, recommend treatment, or establish causation."
    )


# --------------------------------------------------------------------------- #
# JSON Schema export (for reuse by other researchers / languages)
# --------------------------------------------------------------------------- #
_MODELS = {
    "SubjectProfile": SubjectProfile,
    "SourceReference": SourceReference,
    "HealthEvent": HealthEvent,
    "CycleContext": CycleContext,
    "Finding": Finding,
    "DoctorBrief": DoctorBrief,
}


def export_json_schema() -> dict:
    """Return a dict of {entity_name: json_schema} for all core entities."""
    return {name: model.model_json_schema() for name, model in _MODELS.items()}
