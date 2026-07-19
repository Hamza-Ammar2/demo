# Medical Safety

Safety is enforced in code (`cyclebench/safety.py`), not just in prompt instructions.

## Language guard
`find_violations()` scans every generated brief string for **affirmative** diagnostic,
causal, or treatment-recommendation claims (e.g. "you have", "diagnosed with", "is caused by",
"we recommend you take"). The guard is **negation-aware**: explicitly negated or hedged
phrasing ("this does not diagnose", "not caused by", "may be associated with") is allowed.
`assert_brief_safe()` runs on every `DoctorBrief` the pipeline emits; a violation raises
`SafetyViolation` and fails the build/tests — so unsafe language cannot ship silently.

## Structural safeguards
- **Association, not causation.** Findings can only reach `established` / `possible`; the
  statement templates never assert a diagnosis or a mechanism.
- **Provenance-or-silence.** Any asserting finding without supporting events/sources is
  rejected by schema validation (`Finding._provenance_required`).
- **Confounders surfaced, not adjusted.** Overlapping sleep/stress/medication changes are
  attached to findings so a clinician sees alternative explanations.
- **Missing information is a first-class output**, not an omission.
- **The LLM never computes a finding.** It only extracts candidate events (validated against
  schema) and phrases explanations; all numbers come from deterministic code.
- **Every brief carries a disclaimer** and declares its analysis mode.

## Privacy
- De-identified schema (coarse `age_range`, no DOB); sources store pointers/excerpts, not raw
  records. The API store is in-memory and non-persistent.
- Restricted mcPHASES data never leaves the local machine; only aggregate metrics are written.

## Escalation
The product is explicitly framed as appointment-preparation. It does not triage emergencies;
users are directed to seek professional care for medical concerns.
