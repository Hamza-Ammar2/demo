"""Assemble a FoundationRead from intake + optional personal engine findings.

Chat answer formula per card:
  foundation fact + dataset evidence summaries + personal pattern + ask-doctor question
"""

from __future__ import annotations

from typing import Any, Optional

from cyclebench.foundation.io import load_bundle
from cyclebench.foundation.schema import (
    FoundationBundle,
    FoundationCard,
    FoundationRead,
)
from cyclebench.safety import assert_safe


def intake_tags(intake: dict) -> set[str]:
    """Derive match tags from structured intake."""
    tags: set[str] = set()
    for s in intake.get("symptoms") or []:
        t = s.get("type")
        if t:
            tags.add(f"symptom:{t}")
            if t in ("headache", "migraine"):
                tags.add("symptom:headache")
                tags.add("symptom:migraine")
            if t in ("mood", "fatigue", "bloating", "pelvic_pain", "cramps"):
                tags.add("cluster:pcosish")
            if t == "hot_flash":
                tags.add("symptom:hot_flash")
            if t == "weight_change" or t == "fatigue":
                pass
    if intake.get("symptoms"):
        tags.add("has_symptoms")

    timing = intake.get("symptom_timing")
    if timing:
        tags.add(f"timing:{timing}")

    sleep = intake.get("sleep_quality")
    if sleep in ("rough", "bad"):
        tags.add("sleep:poor")

    cstat = intake.get("contraception_status")
    if cstat:
        tags.add(f"contraception:{cstat}")
    form = (intake.get("contraception_formulation") or "").lower()
    if any(k in form for k in ("combined", "ring", "patch", "estrogen")):
        tags.add("contraception:estrogen")
        tags.add("contraception:combined_pill")
    for key in ("pop", "hormonal_iud", "copper_iud", "implant", "injection"):
        if key.replace("_", " ") in form or key in form:
            tags.add(f"contraception:{key}")

    reg = intake.get("cycle_regularity")
    if reg in ("irregular", "very_irregular", "none"):
        tags.add("cycle:irregular")
        tags.add("cluster:pcosish")
    if reg == "none":
        tags.add("cycle:amenorrhea")

    lp = intake.get("last_period_days_ago")
    if isinstance(lp, (int, float)) and lp >= 90:
        tags.add("last_period:long")
        tags.add("cycle:amenorrhea")

    age = intake.get("age_range") or ""
    if age in ("40-49", "50-59"):
        tags.add("age:midlife")
    if age in ("under-20",):
        tags.add("age:teen")

    if (intake.get("hot_flash_freq") or 0) >= 2:
        tags.add("symptom:hot_flash")
    if (intake.get("night_sweat_freq") or 0) >= 2:
        tags.add("symptom:night_sweat")

    if (
        "age:midlife" in tags
        and (
            "symptom:hot_flash" in tags
            or "symptom:night_sweat" in tags
            or "cycle:irregular" in tags
        )
    ):
        tags.add("vasomotor_or_cycle_change")
        tags.add("model:menopause")

    other = intake.get("other_changes") or []
    if "weight" in other or "symptom:weight_change" in tags:
        tags.add("context:weight")
        tags.add("cluster:pcosish")
    if "stress" in other:
        tags.add("context:stress")
    if "exercise" in other:
        tags.add("context:exercise")
    if "med" in other:
        tags.add("context:med")

    if "cluster:pcosish" in tags and "cycle:irregular" in tags:
        tags.add("model:pcos")

    bw = intake.get("bloodwork")
    if bw in ("fsh_high", "fsh_normal"):
        tags.add("labs:fsh")

    return tags


def _personal_for_association(assoc_id: str, personal_findings: list | None) -> Optional[str]:
    """Attach a personal engine finding ONLY when it clearly belongs to this association.

    No generic fallback — that previously glued unrelated findings (e.g. contraception
    change rate) onto hormone reference cards like estradiol/NHANES.
    """
    if not personal_findings:
        return None
    keys = {
        "assoc.estrogen_migraine": ["contraception", "migraine", "headache"],
        "assoc.contraception_change_window": ["contraception", "after the recorded", "frequency"],
        "assoc.sleep_headache": ["sleep"],
        "assoc.sleep_mood": ["sleep"],
        "assoc.sleep_fatigue": ["sleep"],
        "assoc.headache_menstrual": ["cycle", "phase", "luteal", "menstrual", "window"],
        "assoc.migraine_luteal": ["cycle", "phase", "luteal", "window"],
        "assoc.cramps_menstrual": ["cramp", "cycle", "phase"],
        "assoc.pop_headaches": ["headache", "phase", "cycle"],
    }
    needles = keys.get(assoc_id)
    if not needles:
        return None
    for f in personal_findings:
        stmt = f.statement if hasattr(f, "statement") else str(f.get("statement", ""))
        low = stmt.lower()
        if any(n in low for n in needles):
            return stmt
    return None


def _runtime_model_signals(intake: dict, tags: set[str]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    if "model:pcos" in tags:
        try:
            from cyclebench.model.tasks import predict_task, task_to_finding
            # Map coarse intake → sparse PCOS features the model knows
            feats = {
                "Cycle(R/I)": 4 if "cycle:irregular" in tags else 2,
                "Weight gain(Y/N)": 1 if "context:weight" in tags else 0,
                "hair growth(Y/N)": 1 if any(
                    (s.get("type") == "hairsuit") for s in (intake.get("symptoms") or [])
                ) else 0,
                "Skin darkening (Y/N)": 0,
                "Pimples(Y/N)": 1 if any(
                    (s.get("type") == "acne") for s in (intake.get("symptoms") or [])
                ) else 0,
                "Age (yrs)": {"20-29": 25, "30-39": 35, "40-49": 45}.get(
                    intake.get("age_range"), 30
                ),
            }
            pred = predict_task("pcos_risk", feats)
            finding = task_to_finding("pcos_risk", pred)
            assert_safe(finding.statement, where="runtime_pcos")
            signals.append({
                "task": "pcos_risk",
                "association_id": "assoc.model_pcos",
                "probability": pred.get("positive_probability"),
                "statement": finding.statement,
                "explanation": pred.get("explanation", []),
            })
        except Exception:
            pass

    if "model:menopause" in tags:
        try:
            from cyclebench.intake import intake_to_menopause_features, menopause_relevant
            from cyclebench.model.predict import menopause_stage_to_finding, predict_menopause_stage
            if menopause_relevant(intake):
                feats = intake_to_menopause_features(intake)
                pred = predict_menopause_stage(feats)
                finding = menopause_stage_to_finding(pred)
                assert_safe(finding.statement, where="runtime_meno")
                signals.append({
                    "task": "menopause_stage",
                    "association_id": "assoc.model_meno",
                    "predicted_stage": pred.get("predicted_stage"),
                    "confidence": pred.get("confidence"),
                    "statement": finding.statement,
                    "explanation": pred.get("explanation", []),
                    "data_source": "see model metrics (may be synthetic_swan_like)",
                })
        except Exception:
            pass
    return signals


def assemble_read(
    intake: dict,
    personal_findings: list | None = None,
    bundle: FoundationBundle | None = None,
) -> FoundationRead:
    bundle = bundle or load_bundle()
    tags = intake_tags(intake)
    matched_entities: set[str] = set()
    cards: list[FoundationCard] = []
    questions: list[str] = []

    for assoc in bundle.associations:
        if not assoc.match_tags:
            continue
        needed = set(assoc.match_tags)
        overlap = needed & tags
        if not overlap:
            continue
        # If the association is about specific symptoms, require at least one symptom tag match
        symptom_needs = {t for t in needed if t.startswith("symptom:")}
        if symptom_needs and not (symptom_needs & tags):
            continue
        # Estrogen counseling edges require estrogen method
        if "contraception:estrogen" in needed and "contraception:estrogen" not in tags:
            continue
        # Contraception-change window
        if assoc.association_id == "assoc.contraception_change_window":
            if not ({"contraception:changed", "contraception:stopped"} & tags):
                continue
            if "has_symptoms" not in tags:
                continue
        # Amenorrhea edge
        if assoc.association_id == "assoc.amenorrhea_eval" and "cycle:amenorrhea" not in tags:
            continue
        # Midlife composite
        if assoc.association_id == "assoc.midlife_composite" and "age:midlife" not in tags:
            continue
        # PCOS irregular edge needs irregular cycles
        if assoc.association_id == "assoc.irregular_pcos" and "cycle:irregular" not in tags:
            continue
        # Sleep edges need poor sleep
        if "sleep:poor" in needed and "sleep:poor" not in tags:
            continue
        # Context tags (stress/exercise/med/weight) are required when listed
        context_needs = {t for t in needed if t.startswith("context:")}
        if context_needs and not (context_needs & tags):
            continue
        # Explicit contraception method tags beyond estrogen/changed
        method_needs = {
            t for t in needed
            if t.startswith("contraception:") and t.split(":", 1)[1] not in {
                "estrogen", "changed", "stopped", "on_stable", "none",
            }
        }
        if method_needs and not (method_needs & tags):
            continue
        # Timing-sensitive symptom edges: if they list timing tags, require one when present in needed
        timing_needs = {t for t in needed if t.startswith("timing:")}
        if timing_needs and symptom_needs and not (timing_needs & tags):
            # allow match without timing only for non-timing-primary edges; for menstrual headache require timing OR constant
            if assoc.association_id in {
                "assoc.headache_menstrual", "assoc.migraine_luteal", "assoc.mood_luteal",
                "assoc.mood_pmdd", "assoc.pelvic_menstrual", "assoc.pelvic_endo", "assoc.cramps_menstrual",
            }:
                continue
        # Model edges
        if assoc.relation.value == "predicted_by_model" and not (tags & {"model:pcos", "model:menopause"}):
            continue
        # population enrichment: require the symptom
        if assoc.relation.value == "population_enriched_in" and not symptom_needs & tags:
            continue

        matched_entities.add(assoc.subject_id)
        matched_entities.add(assoc.object_id)

        evs = bundle.evidence_for(assoc.association_id)
        # Prefer cohort/reference/model over qa for display order
        order = {"cohort_rate": 0, "reference_range": 1, "model_signal": 2, "guideline_seed": 3, "qa_citation": 4}
        evs = sorted(evs, key=lambda e: order.get(e.evidence_type.value, 9))
        summaries = [e.summary_sentence for e in evs if e.evidence_type.value != "qa_citation"][:3]
        for s in summaries:
            assert_safe(s, where=f"evidence:{assoc.association_id}")

        fact = assoc.talking_point or f"{assoc.subject_id} {assoc.relation.value} {assoc.object_id}"
        assert_safe(fact, where=f"fact:{assoc.association_id}")
        if assoc.ask_doctor:
            assert_safe(assoc.ask_doctor, where=f"ask:{assoc.association_id}")
            if assoc.ask_doctor not in questions:
                questions.append(assoc.ask_doctor)

        personal = _personal_for_association(assoc.association_id, personal_findings)
        cards.append(FoundationCard(
            association_id=assoc.association_id,
            title=assoc.title or assoc.association_id,
            foundation_fact=fact,
            evidence_summaries=summaries,
            personal_pattern=personal,
            ask_doctor=assoc.ask_doctor,
            source=assoc.source,
            datasets=sorted({e.dataset.value for e in evs}),
            strength_prior=assoc.strength_prior.value,
            relation=assoc.relation.value,
        ))

    # Deduplicate cards by association_id; prefer higher strength
    strength_rank = {"high": 0, "medium": 1, "low": 2}
    cards.sort(key=lambda c: (strength_rank.get(c.strength_prior, 9), c.association_id))
    seen = set()
    uniq = []
    for c in cards:
        if c.association_id in seen:
            continue
        seen.add(c.association_id)
        uniq.append(c)
    cards = uniq[:12]  # keep the read focused

    model_signals = _runtime_model_signals(intake, tags)

    missing = []
    if "has_symptoms" in tags and "timing:before_period" not in tags and "timing:during_period" not in tags and "timing:constant" not in tags:
        missing.append("Symptom timing relative to your period would sharpen cycle-linked associations.")
    if "has_symptoms" in tags and not any(t.startswith("contraception:") for t in tags):
        missing.append("Contraception status/type helps evaluate medication-adjustment windows.")
    if "age:midlife" not in tags and "cluster:pcosish" in tags and "cycle:irregular" in tags:
        missing.append("Age range helps choose between midlife-transition vs reproductive-age workup paths.")

    return FoundationRead(
        matched_entity_ids=sorted(matched_entities),
        cards=cards,
        doctor_questions=questions[:8],
        model_signals=model_signals,
        missing_prompts=missing,
        foundation_version=bundle.version,
    )
