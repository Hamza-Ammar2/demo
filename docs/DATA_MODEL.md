# CycleBench Data Model (schema v0.1.0)

A versioned, reusable schema for longitudinal hormonal-health cases. Implemented as
Pydantic v2 (`cyclebench/schema.py`); language-neutral JSON Schema in `docs/schema/*.json`.

## Entities

### SubjectProfile
De-identified subject. `subject_id`, coarse `age_range` (never a DOB), `life_stage`,
`contraception_status` / `contraception_formulation`, `menopause_status`, `hrt_status`,
`timezone`, `source_quality`.

### SourceReference
Provenance anchor. `source_id`, `source_type` (voice/text/diary/wearable/laboratory/
**synthetic_fixture**), `excerpt` (pointer, not raw PII), `observed_at`, `confidence`,
`provenance_status`.

### HealthEvent
One dated observation. Key fields: `event_type` (17 types incl. symptom, menstrual_onset,
medication/contraception changes, sleep/stress/wearable/glucose/hormone/lab measurements),
`start`/`end`, `date_precision` (exact/day/approximate/range/unknown), `value`+`unit`,
`severity` (0–10), `certainty`, **`evidence_class`** (measured/documented/patient_reported/
inferred), `source_id`, `observed_at` (drives causal leakage checks), `cycle_id`,
`medication_context`. Validates that ranges have both ends and end ≥ start.

### CycleContext
A menstrual cycle + phase estimate. `period_onset`, `next_known_onset`, `estimated_phase`,
`phase_confidence`, **`analysis_mode`** (mandatory), `used_future_data`. **Validation:**
causal mode forbids `next_known_onset` and `used_future_data=True` (fails loudly).

### Finding
A deterministic analytical result. `title`, `statement` (association-only), `finding_type`,
`supporting_/contradicting_/confounder_event_ids`, `source_ids`, `strength`, `method`,
`analysis_mode`, **`establishment`** (established/possible/not_established/missing),
`limitation`, `metrics` (the transparent numbers). **Validation:** any *asserting* finding
(established/possible) must carry provenance — the anti-hallucination gate.

### DoctorBrief
Appointment-ready output: `opening_statement`, `strongest_findings`, `unresolved_questions`,
the four buckets (`established`/`possible`/`not_established`/`missing`), `finding_ids`,
`disclaimer`, `analysis_mode`.

## Evidence & establishment ladders
- **evidence_class**: measured ▸ documented ▸ patient_reported ▸ inferred.
- **establishment**: established ▸ possible ▸ not_established ▸ missing.

These let a clinician see, per claim, both *how strong the source is* and *how far the
system is willing to commit* — never beyond "possible / association".
