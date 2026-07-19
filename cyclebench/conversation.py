"""Data-driven conversation plan.

The chat's questions are not arbitrary — each one maps to something the corpus/models
actually use, and the ordering follows the trained menopause-stage model's feature
importances plus the deterministic engine's needs. This is what makes the intake
"based on the datasets/models" rather than a hand-written script.

Every question carries `grounding` metadata (which model feature it feeds, its
importance, and any mcPHASES cohort signal) so the UI can show *why* it's asked.
"""

from __future__ import annotations

from cyclebench.reference import load_reference, symptom_cohort_context

# Canonical intake questions. `maps_to` links each to model features ("meno:<feat>")
# or the deterministic engine ("engine"). `requires` gates on prior answers.
QUESTION_SPECS: list[dict] = [
    {
        "id": "symptoms", "mode": "multi",
        "prompt": "What have you been feeling? Tap all that fit, then continue.",
        "chips": [
            ["Headaches / migraine", "headache"], ["Cramps", "cramps"],
            ["Pelvic pain", "pelvic_pain"], ["Fatigue", "fatigue"],
            ["Brain fog", "brain_fog"], ["Mood changes", "mood"],
            ["Bloating", "bloating"], ["Sore breasts", "sore_breasts"],
            ["Nausea", "nausea"],
        ],
        "maps_to": ["engine", "mcphases:symptoms"],
        "why": "Symptoms are matched against real mcPHASES cycle-phase base rates.",
    },
    {
        "id": "severity", "mode": "single",
        "prompt": "At their worst, how intense?",
        "chips": [["Mild", "mild"], ["Moderate", "moderate"], ["Severe", "severe"]],
        "requires_symptoms": True,
        "maps_to": ["engine"],
        "why": "Severity drives which findings the engine treats as notable.",
    },
    {
        "id": "symptom_timing", "mode": "single",
        "prompt": "When do they tend to hit?",
        "chips": [
            ["Right before my period", "before_period"],
            ["During my period", "during_period"],
            ["After it ends", "after_period"],
            ["Mid-cycle / around ovulation", "mid_cycle"],
            ["All the time / no pattern", "constant"],
        ],
        "requires_symptoms": True,
        "maps_to": ["engine"],
        "why": "Timing lets us compare your pattern to the cohort's phase clustering.",
    },
    {
        "id": "last_period", "mode": "single",
        "prompt": "When did your last period start?",
        "chips": [
            ["In the last few days", 3], ["1–2 weeks ago", 10],
            ["About 3 weeks ago", 21], ["4–8 weeks ago", 42],
            ["Over 2 months ago", 90], ["Not sure", None],
        ],
        "maps_to": ["engine", "meno:amenorrhea_months"],
    },
    {
        "id": "cycle_regularity", "mode": "single",
        "prompt": "Are your cycles usually regular?",
        "chips": [
            ["Regular, ~monthly", "regular"], ["A bit irregular", "irregular"],
            ["Very unpredictable", "very_irregular"], ["No periods right now", "none"],
        ],
        "maps_to": ["meno:cycle_irregularity"],
    },
    {
        "id": "sleep", "mode": "single",
        "prompt": "How's your sleep been this past week?",
        "chips": [["Fine", "ok"], ["A bit rough", "rough"], ["Really bad", "bad"]],
        "maps_to": ["meno:sleep_disturbance", "engine"],
    },
    {
        "id": "hot_flashes", "mode": "single",
        "prompt": "Any hot flashes lately?",
        "chips": [["None", 0], ["Occasionally", 2], ["Often", 4], ["Many a day", 6]],
        "maps_to": ["meno:hot_flash_freq"],
    },
    {
        "id": "night_sweats", "mode": "single",
        "prompt": "Night sweats?",
        "chips": [["None", 0], ["Sometimes", 2], ["Frequently", 4]],
        "maps_to": ["meno:night_sweat_freq"],
    },
    {
        "id": "contraception_status", "mode": "single",
        "prompt": "Any contraception right now?",
        "chips": [
            ["None", "none"], ["Yes, no recent changes", "on_stable"],
            ["Changed it recently", "changed"], ["Recently stopped", "stopped"],
        ],
        "maps_to": ["engine"],
    },
    {
        "id": "contraception_type", "mode": "single",
        "prompt": "Which kind is it (or was it)?",
        "chips": [
            ["Combined pill", "combined_pill"], ["Progestogen-only pill", "pop"],
            ["Hormonal IUD", "hormonal_iud"], ["Copper IUD", "copper_iud"],
            ["Implant", "implant"], ["Injection", "injection"],
            ["Ring / patch", "ring_patch"], ["Not sure", "unknown"],
        ],
        "requires": {"contraception_status": ["on_stable", "changed", "stopped"]},
        "maps_to": ["engine", "safety:migraine_estrogen"],
        "why": "Combined-estrogen methods with migraine-with-aura are worth flagging to a clinician.",
    },
    {
        "id": "contraception_when", "mode": "single",
        "prompt": "Roughly when did that change?",
        "chips": [
            ["In the last month", 25], ["1–3 months ago", 60],
            ["3–6 months ago", 135], ["Over 6 months ago", 220],
        ],
        "requires": {"contraception_status": ["changed", "stopped"]},
        "maps_to": ["engine"],
    },
    {
        "id": "age", "mode": "single",
        "prompt": "Which age range are you in?",
        "chips": [
            ["Under 20", "under-20"], ["20s", "20-29"], ["30s", "30-39"],
            ["40s", "40-49"], ["50+", "50-59"],
        ],
        "maps_to": ["meno:age_years"],
    },
    {
        "id": "bloodwork", "mode": "single",
        "prompt": "Do you have recent hormone bloodwork (FSH/estradiol)? (optional)",
        "chips": [
            ["No / not sure", "none"], ["Yes, FSH was high", "fsh_high"],
            ["Yes, FSH was normal", "fsh_normal"],
        ],
        "maps_to": ["meno:fsh_miu_ml", "meno:estradiol_pg_ml"],
        "why": "FSH is the 2nd-strongest menopause-stage signal in the model.",
    },
    {
        "id": "other_changes", "mode": "multi",
        "prompt": "Anything else changed lately? Tap any — or skip.",
        "chips": [
            ["New medication", "med"], ["Lots of stress", "stress"],
            ["Weight change", "weight"], ["New exercise", "exercise"],
            ["Big diet change", "diet"], ["Nothing else", "none"],
        ],
        "maps_to": ["engine"],
    },
]

# Base questions that should always come first (needed by everything downstream).
_BASE_FIRST = ["symptoms", "severity", "symptom_timing", "last_period"]


def _importance_map() -> dict[str, float]:
    ref = load_reference()
    out: dict[str, float] = {}
    for fi in ref.get("menopause", {}).get("feature_importances", []):
        out[f"meno:{fi['feature']}"] = float(fi.get("importance", 0.0))
    return out


def build_question_plan() -> dict:
    """Return the ordered, grounded question plan the frontend renders."""
    imp = _importance_map()

    def score(q: dict) -> float:
        return max((imp.get(m, 0.0) for m in q.get("maps_to", [])), default=0.0)

    base = [q for q in QUESTION_SPECS if q["id"] in _BASE_FIRST]
    base.sort(key=lambda q: _BASE_FIRST.index(q["id"]))
    rest = [q for q in QUESTION_SPECS if q["id"] not in _BASE_FIRST]
    rest.sort(key=score, reverse=True)  # most model-informative first

    ordered = base + rest
    plan = []
    for q in ordered:
        item = {k: q[k] for k in ("id", "mode", "prompt", "chips") if k in q}
        item["maps_to"] = q.get("maps_to", [])
        item["importance"] = round(score(q), 4)
        if q.get("requires"):
            item["requires"] = q["requires"]
        if q.get("requires_symptoms"):
            item["requires_symptoms"] = True
        if q.get("why"):
            item["why"] = q["why"]
        plan.append(item)

    ref = load_reference()
    grounding = {
        "menopause_top_features": ref.get("menopause", {}).get("feature_importances", [])[:5],
        "menopause_data_source": ref.get("menopause", {}).get("data_source"),
        "mcphases": ref.get("sources", {}).get("mcphases"),
        "symptoms_with_cohort_signal": [
            s for s in ["headache", "cramps", "mood", "bloating", "sore_breasts"]
            if (symptom_cohort_context(s) or {}).get("significant")
        ],
    }
    return {"questions": plan, "grounding": grounding}
