"""Seed the medical foundation graph from guideline/textbook associations.

This is the bootstrap layer. Datasets do NOT invent these edges — they attach Evidence.
Aim: ~100+ entities, ~80+ associations covering core women's hormonal health.
"""

from __future__ import annotations

from cyclebench.foundation.schema import (
    Association,
    AssociationStatus,
    Entity,
    EntityKind,
    FoundationBundle,
    Relation,
    StrengthPrior,
)


def _e(eid, kind, label, synonyms=None, description="", notes="", not_a_diagnosis=False):
    return Entity(
        entity_id=eid, kind=EntityKind(kind), label=label,
        synonyms=synonyms or [], description=description, notes=notes,
        not_a_diagnosis=not_a_diagnosis,
    )


def _a(
    aid, subj, obj, relation, source, *,
    title="", talking_point="", ask_doctor="", match_tags=None,
    strength=StrengthPrior.medium, scope="general", caveats="",
    directionality="bidirectional", status=AssociationStatus.seed,
):
    return Association(
        association_id=aid, subject_id=subj, object_id=obj,
        relation=Relation(relation), source=source, title=title,
        talking_point=talking_point, ask_doctor=ask_doctor,
        match_tags=match_tags or [], strength_prior=strength,
        population_scope=scope, caveats=caveats,
        directionality=directionality, status=status,
    )


def build_seed_entities() -> list[Entity]:
    ents: list[Entity] = []

    # --- symptoms ---
    symptoms = [
        ("sym.headache", "headache", ["headaches"], "Head pain of any pattern"),
        ("sym.migraine", "migraine", ["severe migraine", "migraines"], "Migraine-pattern headache"),
        ("sym.cramps", "cramps", ["period pain", "dysmenorrhea"], "Menstrual cramps"),
        ("sym.pelvic_pain", "pelvic pain", ["pelvic discomfort"], "Pelvic pain"),
        ("sym.fatigue", "fatigue", ["tiredness", "exhaustion"], "Low energy / fatigue"),
        ("sym.brain_fog", "brain fog", ["foggy thinking"], "Cognitive fog / difficulty concentrating"),
        ("sym.mood", "mood changes", ["mood swings", "irritability"], "Mood lability or low mood"),
        ("sym.bloating", "bloating", [], "Abdominal bloating"),
        ("sym.sore_breasts", "breast tenderness", ["sore breasts"], "Cyclical breast tenderness"),
        ("sym.nausea", "nausea", [], "Nausea"),
        ("sym.hot_flash", "hot flash", ["hot flashes", "vasomotor"], "Sudden heat / flushing"),
        ("sym.night_sweat", "night sweats", [], "Night-time sweating"),
        ("sym.acne", "acne", [], "Acne / skin breakouts"),
        ("sym.hairsuit", "excess hair growth", ["hirsutism", "hair growth"], "Androgen-related hair growth discussion cue"),
        ("sym.weight_change", "weight change", ["weight gain"], "Recent weight change"),
        ("sym.low_libido", "low libido", [], "Reduced sexual desire"),
        ("sym.spotting", "spotting", ["breakthrough bleeding"], "Intermenstrual spotting"),
        ("sym.heavy_bleed", "heavy bleeding", ["menorrhagia"], "Heavy menstrual bleeding"),
    ]
    for eid, label, syn, desc in symptoms:
        ents.append(_e(eid, "symptom", label, syn, desc))

    # --- cycle phases / constructs ---
    for eid, label, desc in [
        ("phase.menstrual", "menstrual phase", "Bleeding / early cycle window"),
        ("phase.follicular", "follicular phase", "Post-menstrual follicular window"),
        ("phase.fertility", "fertility / periovulatory", "Around ovulation"),
        ("phase.luteal", "luteal phase", "Post-ovulatory luteal window"),
        ("cycle.irregular", "irregular cycles", "Oligomenorrhea / unpredictable cycles"),
        ("cycle.regular", "regular cycles", "Approximately monthly cycles"),
        ("cycle.amenorrhea", "amenorrhea", "Absence of menses (discussion cue)"),
        ("cycle.short", "short cycles", "Cycles shorter than typical"),
        ("cycle.long", "long cycles", "Cycles longer than typical"),
    ]:
        ents.append(_e(eid, "cycle_phase" if eid.startswith("phase.") else "other", label, description=desc))

    # --- life stages ---
    for eid, label in [
        ("life.adolescence", "adolescence"),
        ("life.reproductive", "reproductive years"),
        ("life.early_peri", "early perimenopause"),
        ("life.late_peri", "late perimenopause"),
        ("life.postmeno", "postmenopause"),
        ("life.pregnancy", "pregnancy"),
        ("life.postpartum", "postpartum"),
    ]:
        ents.append(_e(eid, "life_stage", label))

    # --- hormones / analytes ---
    for eid, label, kind, desc in [
        ("horm.estradiol", "estradiol (E2)", "hormone", "Circulating estradiol"),
        ("horm.progesterone", "progesterone", "hormone", "Circulating / metabolite progesterone"),
        ("horm.fsh", "FSH", "hormone", "Follicle-stimulating hormone"),
        ("horm.lh", "LH", "hormone", "Luteinizing hormone"),
        ("horm.amh", "AMH", "hormone", "Anti-Müllerian hormone"),
        ("horm.tsh", "TSH", "hormone", "Thyroid-stimulating hormone"),
        ("horm.shbg", "SHBG", "hormone", "Sex hormone-binding globulin"),
        ("horm.prolactin", "prolactin", "hormone", "Prolactin"),
        ("horm.testosterone", "testosterone", "hormone", "Total / free testosterone context"),
        ("anal.glucose", "glucose", "analyte", "Fasting / plasma glucose"),
        ("anal.insulin", "insulin", "analyte", "Fasting insulin"),
        ("anal.hba1c", "HbA1c", "analyte", "Glycated hemoglobin"),
        ("anal.t4", "free T4", "analyte", "Thyroxine"),
        ("anal.t3", "free T3", "analyte", "Triiodothyronine"),
    ]:
        ents.append(_e(eid, kind, label, description=desc))

    # --- interventions ---
    for eid, label, syn in [
        ("int.combined_pill", "combined oral contraceptive", ["combined pill", "COC", "estrogen pill"]),
        ("int.pop", "progestogen-only pill", ["POP", "mini pill"]),
        ("int.hormonal_iud", "hormonal IUD", ["Mirena-type IUD", "LNG-IUD"]),
        ("int.copper_iud", "copper IUD", ["Cu-IUD"]),
        ("int.implant", "contraceptive implant", ["implant"]),
        ("int.injection", "contraceptive injection", ["DMPA", "Depo"]),
        ("int.ring_patch", "ring or patch", ["vaginal ring", "patch"]),
        ("int.hrt", "menopausal hormone therapy", ["HRT", "MHT"]),
        ("int.none", "no contraception", ["none"]),
    ]:
        ents.append(_e(eid, "intervention", label, syn))

    # --- confounders ---
    for eid, label, desc in [
        ("ctx.sleep", "sleep disturbance", "Poor or fragmented sleep"),
        ("ctx.stress", "stress", "High psychological stress"),
        ("ctx.bmi", "BMI / body weight context", "Weight / adiposity context"),
        ("ctx.exercise", "exercise change", "New or intense exercise"),
        ("ctx.diet", "diet change", "Major dietary shift"),
        ("ctx.med_change", "medication change", "New / changed medication"),
        ("ctx.contraception_change", "contraception change window", "Start/stop/switch of contraception"),
        ("ctx.iron", "iron status context", "Possible iron deficiency discussion cue"),
    ]:
        ents.append(_e(eid, "confounder", label, description=desc))

    # --- discussion clusters (NOT diagnoses) ---
    clusters = [
        ("cluster.pcos", "PCOS discussion cluster",
         "Irregular cycles + androgenic / metabolic cues clinicians may explore. Not a diagnosis."),
        ("cluster.endo", "Endometriosis-symptom discussion cluster",
         "Cyclical pelvic pain patterns clinicians may explore. Evidence-light in this graph; not a diagnosis."),
        ("cluster.menopause", "Menopause-transition discussion cluster",
         "Midlife cycle change + vasomotor cues. Stage category talk only — not a home diagnosis."),
        ("cluster.thyroid", "Thyroid discussion cluster",
         "Fatigue/mood/cycle change with thyroid labs context. Not a diagnosis."),
        ("cluster.anemia", "Anemia / heavy-bleed discussion cluster",
         "Heavy bleeding + fatigue cue. Not a diagnosis."),
        ("cluster.pmdd", "Premenstrual mood discussion cluster",
         "Luteal-window mood changes clinicians may explore. Not a diagnosis."),
    ]
    for eid, label, notes in clusters:
        ents.append(_e(eid, "cluster", label, notes=notes, not_a_diagnosis=True,
                       description=notes))

    # --- other useful ---
    ents += [
        _e("other.aura", "other", "migraine aura", ["visual aura"], "Visual/sensory aura with headache"),
        _e("other.follicle_count", "other", "ovarian follicle count", [], "Ultrasound follicle count context"),
        _e("other.endometrium", "other", "endometrial thickness", [], "Ultrasound endometrium context"),
        _e("model.pcos_risk", "other", "PCOS-risk model signal", [], "Factory model output slot"),
        _e("model.meno_stage", "other", "menopause-stage model signal", [], "Factory/model output slot"),
        _e("model.hormonal_state", "other", "hormonal-state model signal", [], "mcPHASES-trained phase model slot"),
    ]
    return ents


def build_seed_associations() -> list[Association]:
    A: list[Association] = []

    # ===== Cycle phase ↔ symptoms =====
    phase_sym = [
        ("assoc.headache_menstrual", "sym.headache", "phase.menstrual", "high",
         "Headaches near menses",
         "Headaches that cluster in the menstrual window are a commonly recognized timing pattern.",
         "My headaches seem tied to my period — does that timing matter?",
         ["symptom:headache", "symptom:migraine", "timing:before_period", "timing:during_period"]),
        ("assoc.migraine_luteal", "sym.migraine", "phase.luteal", "medium",
         "Migraine in the luteal window",
         "Some people notice migraine load rise in the late luteal / premenstrual window.",
         "Do my migraine days line up with the second half of my cycle?",
         ["symptom:migraine", "timing:before_period"]),
        ("assoc.cramps_menstrual", "sym.cramps", "phase.menstrual", "high",
         "Cramps and menses",
         "Cramps commonly concentrate in the menstrual phase; severity and sudden change still matter clinically.",
         "How heavy or disruptive are my cramps across cycles?",
         ["symptom:cramps", "timing:during_period"]),
        ("assoc.mood_luteal", "sym.mood", "phase.luteal", "medium",
         "Mood changes before the period",
         "Mood shifts in the luteal window are commonly discussed; tracking helps separate cycle-linked from constant patterns.",
         "Are my mood changes mainly in the week before my period?",
         ["symptom:mood", "timing:before_period"]),
        ("assoc.bloating_luteal", "sym.bloating", "phase.luteal", "medium",
         "Bloating around the luteal window",
         "Bloating often co-travels with the late cycle; diet and GI causes still matter.",
         "Is my bloating mostly cyclical?",
         ["symptom:bloating"]),
        ("assoc.breasts_luteal", "sym.sore_breasts", "phase.luteal", "medium",
         "Breast tenderness and the luteal phase",
         "Cyclical breast tenderness is frequently luteal-predominant.",
         "Is breast tenderness a usual part of my cycle or new?",
         ["symptom:sore_breasts"]),
        ("assoc.fatigue_menstrual", "sym.fatigue", "phase.menstrual", "low",
         "Fatigue around menses",
         "Fatigue can rise around menses but often has multiple contributors (sleep, iron, mood).",
         "Is my fatigue worse on period days, or all month?",
         ["symptom:fatigue"]),
        ("assoc.pelvic_menstrual", "sym.pelvic_pain", "phase.menstrual", "medium",
         "Pelvic pain timed to menses",
         "Pelvic pain that tracks menses is a pattern clinicians often explore further when severe or progressive.",
         "Is my pelvic pain mainly with periods, and is it getting worse over time?",
         ["symptom:pelvic_pain", "timing:during_period", "timing:before_period"]),
    ]
    for aid, s, o, strength, title, tp, ask, tags in phase_sym:
        A.append(_a(
            aid, s, o, "temporally_associated_with",
            "Menstrual physiology / clinical gynecology & headache medicine (textbook-level timing associations).",
            title=title, talking_point=tp, ask_doctor=ask, match_tags=tags,
            strength=StrengthPrior(strength), scope="reproductive-age",
            caveats="Timing associations are not diagnoses; many non-cycle causes exist.",
        ))

    # ===== Contraception =====
    A.append(_a(
        "assoc.estrogen_migraine", "int.combined_pill", "sym.migraine",
        "counseling_relevant_with",
        "UKMEC / CDC MEC style guidance: combined hormonal contraception and migraine (esp. with aura) is a counseling point.",
        title="Hormonal contraception and headaches",
        talking_point=(
            "Headaches that started or changed around a combined (estrogen-containing) "
            "contraceptive are a pattern clinicians often review. This does not mean the "
            "method caused the headaches — it means the combination is worth discussing, "
            "especially if headaches come with aura (flashing lights, zigzag vision)."
        ),
        ask_doctor=(
            "Do my headaches change with my contraceptive method, and should we review "
            "whether an estrogen-containing option is still a good fit?"
        ),
        match_tags=["symptom:headache", "symptom:migraine", "contraception:estrogen"],
        strength=StrengthPrior.high, scope="reproductive-age using CHC",
        caveats="Counseling relevance only — not a home diagnosis or instruction to stop medication.",
        directionality="subject->object",
    ))
    A.append(_a(
        "assoc.estrogen_aura", "int.combined_pill", "other.aura",
        "counseling_relevant_with",
        "UKMEC / CDC MEC: migraine with aura is a key counseling modifier for combined hormonal contraception.",
        title="Aura and combined contraception",
        talking_point=(
            "If headaches include aura, clinicians usually want that detail when reviewing "
            "estrogen-containing contraception. Aura presence changes the counseling conversation."
        ),
        ask_doctor="Do my headaches include aura, and does that change contraceptive choices for me?",
        match_tags=["symptom:migraine", "contraception:estrogen", "flag:aura"],
        strength=StrengthPrior.high,
        caveats="User may not know what aura is; ask, don't assume.",
    ))
    for method, label in [
        ("int.ring_patch", "ring/patch"),
        ("int.pop", "progestogen-only pill"),
        ("int.hormonal_iud", "hormonal IUD"),
    ]:
        A.append(_a(
            f"assoc.bleed_{method.split('.')[-1]}", method, "sym.spotting",
            "modulated_by",
            "Contraceptive counseling: bleeding-pattern changes are common after method start/switch.",
            title=f"Bleeding pattern and {label}",
            talking_point=(
                f"Spotting or bleeding-pattern changes after starting or switching a {label} "
                "are commonly discussed in the adjustment window."
            ),
            ask_doctor="Is this bleeding pattern expected for my method, and when should we reassess?",
            match_tags=["symptom:spotting", f"contraception:{method.split('.')[-1]}"],
            strength=StrengthPrior.medium,
        ))

    A.append(_a(
        "assoc.contraception_change_window", "ctx.contraception_change", "sym.headache",
        "modulated_by",
        "General contraceptive counseling: adjustment windows after start/stop/switch.",
        title="Symptoms after a contraception change",
        talking_point=(
            "When contraception is started, stopped, or switched, the body often needs "
            "weeks to months to settle. New or changing symptoms in that window are "
            "worth noting on a timeline — not as proof of cause, but as useful context."
        ),
        ask_doctor=(
            "These symptoms showed up around my contraception change — is that a "
            "plausible association, or should we look elsewhere?"
        ),
        match_tags=["contraception:changed", "contraception:stopped", "has_symptoms"],
        strength=StrengthPrior.medium,
        caveats="Temporal coincidence ≠ causation.",
    ))

    # ===== Confounders =====
    A.append(_a(
        "assoc.sleep_headache", "ctx.sleep", "sym.headache",
        "confounded_by",
        "Headache and sleep medicine: bidirectional association between sleep disruption and headache.",
        title="Sleep and headache overlap",
        talking_point=(
            "Poor sleep and headaches often travel together. Sleep disturbance can "
            "worsen headache load, and headaches can disrupt sleep — so clinicians "
            "usually ask about both rather than treating them as separate mysteries."
        ),
        ask_doctor=(
            "I've had rough sleep alongside the headaches — should we address sleep "
            "as part of this picture?"
        ),
        match_tags=["symptom:headache", "symptom:migraine", "sleep:poor"],
        strength=StrengthPrior.high,
    ))
    A.append(_a(
        "assoc.sleep_mood", "ctx.sleep", "sym.mood", "confounded_by",
        "Sleep and mood are tightly linked in clinical assessment.",
        title="Sleep and mood",
        talking_point="Mood changes alongside poor sleep are a common co-travel pattern clinicians unpack together.",
        ask_doctor="Could sleep be contributing to how I've been feeling emotionally?",
        match_tags=["symptom:mood", "sleep:poor"],
        strength=StrengthPrior.medium,
    ))
    A.append(_a(
        "assoc.sleep_fatigue", "ctx.sleep", "sym.fatigue", "confounded_by",
        "Clinical basics: sleep debt is a primary fatigue contributor.",
        title="Sleep and fatigue",
        talking_point="Fatigue with poor sleep often improves when sleep is addressed — but persistent fatigue still deserves a broader look.",
        ask_doctor="How much of my fatigue lines up with nights of bad sleep?",
        match_tags=["symptom:fatigue", "sleep:poor"],
        strength=StrengthPrior.high,
    ))
    A.append(_a(
        "assoc.stress_mood", "ctx.stress", "sym.mood", "confounded_by",
        "Behavioral medicine: stress and mood covary strongly.",
        title="Stress and mood",
        talking_point="High stress and mood changes commonly overlap; both deserve space in the story you bring to a clinician.",
        ask_doctor="How much stress has been in the background of these mood changes?",
        match_tags=["symptom:mood", "context:stress"],
        strength=StrengthPrior.medium,
    ))

    # ===== Midlife / menopause =====
    A.append(_a(
        "assoc.hotflash_peri", "sym.hot_flash", "cluster.menopause",
        "marker_of",
        "Menopause practice guidelines: vasomotor symptoms are cardinal midlife counseling topics.",
        title="Hot flashes in midlife context",
        talking_point=(
            "In the 40s–50s, hot flashes are commonly discussed as part of the menopause "
            "transition. Timing and severity help a clinician place you on that spectrum — "
            "without a home diagnosis."
        ),
        ask_doctor="Given my age and hot flashes, what would you track next?",
        match_tags=["symptom:hot_flash", "age:midlife"],
        strength=StrengthPrior.high, scope="midlife",
        caveats="Vasomotor symptoms have non-menopausal causes too.",
    ))
    A.append(_a(
        "assoc.nightsweat_peri", "sym.night_sweat", "cluster.menopause",
        "marker_of",
        "Menopause practice guidelines: night sweats as vasomotor counseling topic.",
        title="Night sweats in midlife context",
        talking_point="Night sweats in midlife are often reviewed alongside cycle changes as a menopause-transition cue.",
        ask_doctor="Are my night sweats likely part of a midlife transition pattern?",
        match_tags=["symptom:night_sweat", "age:midlife"],
        strength=StrengthPrior.high, scope="midlife",
    ))
    A.append(_a(
        "assoc.irregular_peri", "cycle.irregular", "cluster.menopause",
        "marker_of",
        "STRAW+10 / menopause staging frameworks: cycle irregularity is a transition marker.",
        title="Cycle irregularity in midlife",
        talking_point="Changing cycle regularity in the 40s–50s is a classic counseling marker of the menopause transition.",
        ask_doctor="Do my cycle changes fit a menopause-transition pattern for my age?",
        match_tags=["cycle:irregular", "age:midlife"],
        strength=StrengthPrior.high, scope="midlife",
    ))
    A.append(_a(
        "assoc.amenorrhea_eval", "cycle.amenorrhea", "life.reproductive",
        "counseling_relevant_with",
        "Gynecology basics: secondary amenorrhea (≈3+ months) warrants clinical evaluation.",
        title="Long gap since last period",
        talking_point=(
            "Going three or more months without a period (when pregnancy is not the "
            "explanation) is something clinicians usually want to know about. There "
            "are many possible reasons; the useful move is a proper history and, if "
            "needed, labs — not guessing at home."
        ),
        ask_doctor="I haven't had a period in a while — what should we check, and how urgently?",
        match_tags=["cycle:amenorrhea", "last_period:long"],
        strength=StrengthPrior.high,
    ))
    A.append(_a(
        "assoc.fsh_meno", "horm.fsh", "cluster.menopause", "marker_of",
        "Reproductive endocrinology: FSH rises across the menopause transition (context-dependent).",
        title="FSH as a midlife marker",
        talking_point="FSH can help contextualize midlife stage in clinic; a single value is not a home diagnosis of menopause.",
        ask_doctor="Would FSH or other labs add useful context for my stage?",
        match_tags=["labs:fsh", "age:midlife"],
        strength=StrengthPrior.medium, scope="midlife",
        caveats="FSH fluctuates; interpretation is clinical.",
    ))
    A.append(_a(
        "assoc.e2_meno", "horm.estradiol", "cluster.menopause", "marker_of",
        "Reproductive endocrinology: estradiol declines across the transition (variable).",
        title="Estradiol in midlife context",
        talking_point="Estradiol levels can inform midlife assessment but vary widely cycle-to-cycle before menopause.",
        ask_doctor="Do my estradiol results mean anything for symptoms I'm having?",
        match_tags=["labs:estradiol", "age:midlife"],
        strength=StrengthPrior.medium, scope="midlife",
    ))
    A.append(_a(
        "assoc.model_meno", "model.meno_stage", "cluster.menopause",
        "predicted_by_model",
        "CycleBench menopause-stage model (research estimate).",
        title="Menopause-stage model signal",
        talking_point="A research model can estimate a stage category from age/symptoms/labs — association only, not a clinical diagnosis.",
        ask_doctor="How should we interpret this stage estimate alongside my history?",
        match_tags=["model:menopause"],
        strength=StrengthPrior.medium, status=AssociationStatus.model_linked,
        caveats="May be trained on synthetic SWAN-like data until real SWAN is loaded.",
    ))

    # ===== PCOS cluster =====
    A.append(_a(
        "assoc.irregular_pcos", "cycle.irregular", "cluster.pcos", "marker_of",
        "Reproductive endocrinology: oligomenorrhea is a core PCOS counseling cue (many causes).",
        title="Irregular cycles and related symptoms",
        talking_point=(
            "Irregular cycles plus symptoms like weight change, excess hair growth, "
            "acne, or pelvic discomfort are a cluster clinicians may explore further "
            "(including hormonal workup). Many things can cause irregular cycles — "
            "this is a reason for a conversation, not a label."
        ),
        ask_doctor="My cycles are irregular and I've also noticed other changes — what evaluations would you consider?",
        match_tags=["cycle:irregular", "cluster:pcosish"],
        strength=StrengthPrior.high,
        caveats="Irregular cycles ≠ PCOS.",
    ))
    for aid, subj, title, tp, tags in [
        ("assoc.amh_pcos", "horm.amh", "AMH in PCOS-cluster context",
         "AMH is sometimes discussed in PCOS workups; elevated values are contextual, not diagnostic alone.",
         ["labs:amh"]),
        ("assoc.lh_pcos", "horm.lh", "LH in PCOS-cluster context",
         "LH (and LH/FSH context) may appear in hormonal panels when irregular cycles are evaluated.",
         ["labs:lh"]),
        ("assoc.hairsuit_pcos", "sym.hairsuit", "Excess hair growth cue",
         "Excess hair growth with irregular cycles is a hyperandrogenic cue clinicians may explore — many causes exist.",
         ["symptom:hairsuit", "cluster:pcosish"]),
        ("assoc.acne_pcos", "sym.acne", "Acne with irregular cycles",
         "Acne plus irregular cycles can be part of a broader hormonal conversation.",
         ["symptom:acne", "cluster:pcosish"]),
        ("assoc.weight_pcos", "sym.weight_change", "Weight change with irregular cycles",
         "Weight change alongside irregular cycles is useful context, not a diagnosis.",
         ["symptom:weight_change", "cluster:pcosish", "context:weight"]),
        ("assoc.follicle_pcos", "other.follicle_count", "Follicle count context",
         "Ultrasound follicle counts are used in clinical PCOS criteria frameworks — imaging belongs in clinic, not apps.",
         ["labs:follicles"]),
    ]:
        A.append(_a(
            aid, subj, "cluster.pcos", "marker_of",
            "PCOS clinical frameworks (Rotterdam-style counseling cues) — discussion only.",
            title=title, talking_point=tp, ask_doctor="Does this fit a hormonal pattern worth testing?",
            match_tags=tags, strength=StrengthPrior.medium,
            caveats="Cluster markers ≠ diagnosis.",
        ))
    A.append(_a(
        "assoc.model_pcos", "model.pcos_risk", "cluster.pcos", "predicted_by_model",
        "CycleBench PCOS-risk model trained on Kaggle Kerala clinic cohort (n=541).",
        title="PCOS-risk model signal",
        talking_point=(
            "A research model can estimate how similar a feature pattern is to a labeled "
            "PCOS clinic cohort. That is a statistical association — not a diagnosis."
        ),
        ask_doctor="How should we use this screening-style signal in my clinical workup?",
        match_tags=["model:pcos", "cluster:pcosish"],
        strength=StrengthPrior.medium, status=AssociationStatus.model_linked,
        scope="clinic-PCOS-cohort (Kerala)",
        caveats="Single-region, cross-sectional, not externally validated as a device.",
    ))

    # ===== Endometriosis-ish (low prior) =====
    A.append(_a(
        "assoc.pelvic_endo", "sym.pelvic_pain", "cluster.endo", "marker_of",
        "Gynecology: severe cyclical pelvic pain is a reason to evaluate; many etiologies.",
        title="Cyclical pelvic pain — discussion cue",
        talking_point=(
            "Severe pelvic pain that tracks the cycle is something clinicians take seriously. "
            "Endometriosis is one of several possibilities — this graph marks it as a "
            "low-prior discussion cluster, not a conclusion."
        ),
        ask_doctor="My pelvic pain is severe and cyclical — what should we evaluate?",
        match_tags=["symptom:pelvic_pain", "timing:during_period", "timing:before_period"],
        strength=StrengthPrior.low,
        caveats="Evidence-light edge in this foundation; do not over-call.",
    ))

    # ===== Thyroid / metabolic (modest) =====
    A.append(_a(
        "assoc.tsh_fatigue", "horm.tsh", "sym.fatigue", "counseling_relevant_with",
        "Endocrinology basics: thyroid labs are often considered in unexplained fatigue.",
        title="Thyroid labs and fatigue",
        talking_point="Persistent fatigue sometimes prompts thyroid testing; abnormal TSH needs clinical interpretation.",
        ask_doctor="Would thyroid labs be reasonable given my fatigue?",
        match_tags=["symptom:fatigue", "labs:tsh"],
        strength=StrengthPrior.medium,
    ))
    A.append(_a(
        "assoc.tsh_cluster", "horm.tsh", "cluster.thyroid", "marker_of",
        "Thyroid evaluation frameworks.",
        title="TSH in thyroid discussion cluster",
        talking_point="TSH is a common entry lab when thyroid contribution to symptoms is considered.",
        ask_doctor="Do my thyroid labs explain any of these symptoms?",
        match_tags=["labs:tsh"],
        strength=StrengthPrior.medium,
    ))
    A.append(_a(
        "assoc.glucose_metabolic", "anal.glucose", "cluster.pcos", "counseling_relevant_with",
        "Metabolic–reproductive overlap counseling (insulin resistance context).",
        title="Glucose context with reproductive symptoms",
        talking_point="Glucose/metabolic labs sometimes sit alongside reproductive evaluations; they do not label a syndrome alone.",
        ask_doctor="Should we look at metabolic labs as part of this picture?",
        match_tags=["labs:glucose", "cluster:pcosish"],
        strength=StrengthPrior.low,
    ))

    # ===== Heavy bleed / anemia cue =====
    A.append(_a(
        "assoc.heavy_fatigue", "sym.heavy_bleed", "cluster.anemia", "marker_of",
        "Gynecology / hematology: heavy menses + fatigue prompts iron evaluation discussion.",
        title="Heavy bleeding and fatigue",
        talking_point="Heavy periods with fatigue are a classic reason clinicians check iron status — many other causes of fatigue exist.",
        ask_doctor="Could heavy bleeding be contributing to my fatigue, and should we check iron?",
        match_tags=["symptom:heavy_bleed", "symptom:fatigue"],
        strength=StrengthPrior.medium,
    ))

    # ===== PMDD-ish mood cluster =====
    A.append(_a(
        "assoc.mood_pmdd", "sym.mood", "cluster.pmdd", "marker_of",
        "Premenstrual disorders clinical frameworks (discussion only).",
        title="Premenstrual mood pattern",
        talking_point="Mood symptoms that reliably appear in the luteal window and ease after menses start are a pattern clinicians may explore.",
        ask_doctor="Do my mood symptoms clear after my period starts?",
        match_tags=["symptom:mood", "timing:before_period"],
        strength=StrengthPrior.medium,
        caveats="Not a PMDD diagnosis.",
    ))

    # ===== Population enrichment placeholders (evidence attaches later) =====
    for sym, phase, col in [
        ("sym.headache", "phase.menstrual", "headaches"),
        ("sym.cramps", "phase.menstrual", "cramps"),
        ("sym.fatigue", "phase.menstrual", "fatigue"),
        ("sym.mood", "phase.menstrual", "moodswing"),
        ("sym.bloating", "phase.menstrual", "bloating"),
        ("sym.sore_breasts", "phase.menstrual", "sorebreasts"),
    ]:
        A.append(_a(
            f"assoc.pop_{col}", sym, phase, "population_enriched_in",
            "mcPHASES aggregate symptom×phase rates (PhysioNet; aggregate only).",
            title=f"Cohort enrichment: {col}",
            talking_point=f"In longitudinal cohorts, reported {col} may concentrate in particular cycle phases.",
            ask_doctor="Does my pattern match what shows up in cycle-phase research cohorts?",
            match_tags=[f"mcphases:{col}", f"symptom:{sym.split('.')[-1]}"],
            strength=StrengthPrior.medium, status=AssociationStatus.seed,
            scope="mcPHASES young-adult menstruators",
            caveats="Small cohort; not personally diagnostic.",
        ))

    # Midlife composite (migrated from knowledge midlife_transition)
    A.append(_a(
        "assoc.midlife_composite", "life.early_peri", "cluster.menopause",
        "counseling_relevant_with",
        "Menopause practice guidelines: vasomotor symptoms + cycle change are cardinal midlife counseling topics.",
        title="Midlife cycle and vasomotor changes",
        talking_point=(
            "In the 40s–50s, changing cycle regularity, hot flashes, and night sweats "
            "are commonly discussed as part of the menopause transition. Timing and "
            "severity help a clinician place you on that spectrum — without a home diagnosis."
        ),
        ask_doctor=(
            "Given my age and these changes, does this sound like a menopause-transition "
            "pattern, and what would you track next?"
        ),
        match_tags=["age:midlife", "vasomotor_or_cycle_change"],
        strength=StrengthPrior.high, scope="midlife",
    ))

    # ===== Extra high-value edges (expand foundation coverage) =====
    extras = [
        ("assoc.nausea_menstrual", "sym.nausea", "phase.menstrual", "temporally_associated_with",
         "Nausea around menses", "Nausea can accompany menstrual days for some people; severe or persistent nausea still deserves a broad look.",
         "Is my nausea mainly on period days?", ["symptom:nausea", "timing:during_period"], "medium"),
        ("assoc.brainfog_luteal", "sym.brain_fog", "phase.luteal", "temporally_associated_with",
         "Brain fog and cycle timing", "Some people notice fogginess in the late cycle; sleep and stress often co-travel.",
         "Does brain fog track a particular part of my cycle?", ["symptom:brain_fog"], "low"),
        ("assoc.libido_luteal", "sym.low_libido", "phase.luteal", "temporally_associated_with",
         "Libido and cycle window", "Desire can vary across the cycle; sudden persistent loss is still worth clinical mention.",
         "Has my libido changed recently or only in one cycle window?", ["symptom:low_libido"], "low"),
        ("assoc.pop_bleed", "int.pop", "sym.spotting", "modulated_by",
         "POP and bleeding pattern", "Progestogen-only pills commonly change bleeding patterns, especially early on.",
         "Is this bleeding pattern expected on my progestogen-only pill?", ["symptom:spotting", "contraception:pop"], "medium"),
        ("assoc.implant_bleed", "int.implant", "sym.spotting", "modulated_by",
         "Implant and bleeding pattern", "Implants often change bleeding — from amenorrhea to irregular spotting.",
         "Is my implant-related bleeding within the expected range?", ["symptom:spotting", "contraception:implant"], "medium"),
        ("assoc.copper_heavy", "int.copper_iud", "sym.heavy_bleed", "modulated_by",
         "Copper IUD and heavier flow", "Copper IUDs are classically associated with heavier menstrual bleeding for some users.",
         "Could my copper IUD be contributing to heavier periods?", ["symptom:heavy_bleed", "contraception:copper_iud"], "medium"),
        ("assoc.hrt_vasomotor", "int.hrt", "sym.hot_flash", "modulated_by",
         "MHT and vasomotor symptoms", "Menopausal hormone therapy is discussed clinically for vasomotor symptoms — decisions belong with a clinician.",
         "Is menopausal hormone therapy something we should discuss for these symptoms?", ["symptom:hot_flash", "age:midlife"], "medium"),
        ("assoc.stress_headache", "ctx.stress", "sym.headache", "confounded_by",
         "Stress and headache", "Stress and headache commonly intensify together; both belong in the history.",
         "How much stress has been sitting alongside these headaches?", ["symptom:headache", "context:stress"], "medium"),
        ("assoc.exercise_cycle", "ctx.exercise", "cycle.irregular", "modulated_by",
         "Exercise change and cycles", "Sudden intense exercise or energy deficit can associate with cycle changes in some people.",
         "Did my cycles change after a big shift in training or fueling?", ["cycle:irregular", "context:exercise"], "low"),
        ("assoc.medchange_symptoms", "ctx.med_change", "sym.headache", "modulated_by",
         "Medication change window", "New medications can temporally overlap new symptoms — worth a timeline, not an assumption of cause.",
         "Did symptoms start around a medication change?", ["has_symptoms", "context:med"], "low"),
        ("assoc.prolactin_amenorrhea", "horm.prolactin", "cycle.amenorrhea", "counseling_relevant_with",
         "Prolactin in amenorrhea workups", "Prolactin is sometimes checked when periods stop; interpretation is clinical.",
         "Should prolactin be part of evaluating my missing periods?", ["cycle:amenorrhea", "labs:prolactin"], "medium"),
        ("assoc.shbg_androgen", "horm.shbg", "cluster.pcos", "marker_of",
         "SHBG in androgen/PCOS-cluster labs", "SHBG can appear in hormonal panels when androgen excess is being explored.",
         "Do my androgen-related labs include SHBG, and what do they mean together?", ["labs:shbg", "cluster:pcosish"], "low"),
        ("assoc.insulin_pcos", "anal.insulin", "cluster.pcos", "counseling_relevant_with",
         "Insulin context with PCOS-cluster cues", "Insulin/metabolic labs sometimes accompany reproductive evaluations.",
         "Should insulin or metabolic labs be part of this workup?", ["labs:insulin", "cluster:pcosish"], "low"),
        ("assoc.hba1c_metabolic", "anal.hba1c", "ctx.bmi", "counseling_relevant_with",
         "HbA1c as metabolic context", "HbA1c provides longer-window glucose context; not a reproductive diagnosis.",
         "Would an HbA1c help round out the metabolic side of this picture?", ["labs:hba1c"], "low"),
        ("assoc.postpartum_mood", "life.postpartum", "sym.mood", "temporally_associated_with",
         "Postpartum mood window", "The postpartum period is a high-attention window for mood changes — seek clinical care promptly for concerning symptoms.",
         "Are these mood changes in a postpartum window, and how urgently should we review them?", ["symptom:mood", "life:postpartum"], "high"),
        ("assoc.adolescence_irregular", "life.adolescence", "cycle.irregular", "temporally_associated_with",
         "Adolescent cycle irregularity", "Cycles can be irregular for years after menarche; persistent severe symptoms still deserve care.",
         "Is this irregularity expected for my age since periods started?", ["cycle:irregular", "age:teen"], "medium"),
    ]
    for aid, s, o, rel, title, tp, ask, tags, strength in extras:
        A.append(_a(
            aid, s, o, rel,
            "Expanded CycleBench foundation seed (guideline/textbook-level associations).",
            title=title, talking_point=tp, ask_doctor=ask, match_tags=tags,
            strength=StrengthPrior(strength),
        ))

    return A


def build_seed_bundle() -> FoundationBundle:
    ents = build_seed_entities()
    assocs = build_seed_associations()
    return FoundationBundle(
        version="v0.1",
        entities=ents,
        associations=assocs,
        evidence=[],  # attached by evidence adapters
    )
