"""CycleBench command-line interface.

    python -m cyclebench.cli demo   [--mode retrospective|causal]
    python -m cyclebench.cli audit
"""

from __future__ import annotations

import argparse
import sys

from cyclebench.engine import compile_case
from cyclebench.fixtures import build_sarah_case
from cyclebench.schema import AnalysisMode

_RULE = "─" * 74


def _print_brief(brief, findings) -> None:
    print(_RULE)
    print("CASE COMPILER — DOCTOR BRIEF   (subject: %s, mode: %s)"
          % (brief.subject_id, brief.analysis_mode.value))
    print(_RULE)
    print("\n30-SECOND OPENING STATEMENT\n")
    print("  " + brief.opening_statement)

    def section(title, items):
        print(f"\n{title}")
        if not items:
            print("  (none)")
        for it in items:
            print(f"  • {it}")

    section("THREE STRONGEST FINDINGS", brief.strongest_findings)
    section("THREE UNRESOLVED QUESTIONS", brief.unresolved_questions)
    section("ESTABLISHED", brief.established)
    section("POSSIBLE", brief.possible)
    section("NOT ESTABLISHED", brief.not_established)
    section("MISSING", brief.missing)

    print("\nFINDING PROVENANCE")
    for f in findings:
        print(f"  [{f.finding_id}] {f.title}")
        print(f"       method={f.method} | mode={f.analysis_mode.value} | "
              f"establishment={f.establishment.value} | strength={f.strength.value}")
        if f.supporting_event_ids:
            print(f"       supporting={f.supporting_event_ids}")
        if f.confounder_event_ids:
            print(f"       confounders={f.confounder_event_ids}")
        if f.metrics:
            print(f"       metrics={f.metrics}")

    print(f"\nDISCLAIMER\n  {brief.disclaimer}")
    print(_RULE)


def cmd_demo(args) -> int:
    mode = AnalysisMode(args.mode)
    case = build_sarah_case()
    result = compile_case(case, mode)
    _print_brief(result.brief, result.findings)
    return 0


def cmd_audit(args) -> int:
    from cyclebench.audit import run_demo_audit
    ok = run_demo_audit()
    return 0 if ok else 1


def cmd_train_models(args) -> int:
    from cyclebench.model.train import train_all
    summary = train_all()
    print(summary)
    return 0 if all(v.get("ok") for v in summary.values()) else 1


def cmd_model_demo(args) -> int:
    """Show multi-source hormonal-state + menopause-stage model outputs with explanations."""
    from cyclebench.model.predict import (
        hormonal_state_to_finding,
        menopause_stage_to_finding,
        predict_hormonal_state,
        predict_menopause_stage,
    )
    from cyclebench.model.train import train_all
    from cyclebench.model.common import MODELS_DIR

    # Ensure checkpoints exist
    if not (MODELS_DIR / "menopause_stage_v0.1.joblib").exists():
        train_all()

    print(_RULE)
    print("LAYER 02 — MULTI-SOURCE HORMONAL STATE + MENOPAUSE STAGE")
    print(_RULE)

    # Example multi-source day (symptom-heavy menstrual-like pattern)
    hs_features = {
        "headaches_ord": 5, "cramps_ord": 5, "fatigue_ord": 4, "sleepissue_ord": 4,
        "moodswing_ord": 3, "stress_ord": 3, "bloating_ord": 4, "sorebreasts_ord": 2,
        "foodcravings_ord": 3, "indigestion_ord": 2, "appetite_ord": 3,
        "sleep_minutes": 320, "sleep_efficiency": 78, "resting_hr": 78,
        "steps_sum": 4000, "stress_score_mean": 70, "wrist_temp_delta": 0.1,
        "glucose_mean": 95, "hrv_rmssd": 25, "is_weekend": 0,
    }
    if (MODELS_DIR / "hormonal_state_v0.1.joblib").exists():
        hs = predict_hormonal_state(hs_features)
        print("\n[1] Multi-source hormonal-state model")
        print(f"    predicted: {hs['predicted_state']}  confidence={hs['confidence']}")
        print(f"    probabilities: {hs['probabilities']}")
        print(f"    explanation: {hs['explanation']}")
        f1 = hormonal_state_to_finding(hs)
        print(f"    brief statement: {f1.statement}")
    else:
        print("\n[1] hormonal_state model unavailable (mcPHASES not present)")

    # Example midlife profile consistent with late perimenopause-like signals
    ms_features = {
        "age_years": 51, "fsh_miu_ml": 42, "estradiol_pg_ml": 28, "shbg_nmol_l": 50,
        "hot_flash_freq": 4, "night_sweat_freq": 3, "sleep_disturbance": 3,
        "cycle_irregularity": 2, "amenorrhea_months": 5, "bmi": 27,
    }
    ms = predict_menopause_stage(ms_features)
    print("\n[2] Menopause-stage model")
    print(f"    predicted: {ms['predicted_stage']}  confidence={ms['confidence']}")
    print(f"    probabilities: {ms['probabilities']}")
    print(f"    explanation: {ms['explanation']}")
    f2 = menopause_stage_to_finding(ms)
    print(f"    brief statement: {f2.statement}")
    print(_RULE)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cyclebench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="print Sarah's doctor brief (offline)")
    p_demo.add_argument("--mode", default="retrospective",
                        choices=["retrospective", "causal"])
    p_demo.set_defaults(func=cmd_demo)

    p_audit = sub.add_parser("audit", help="run the leakage-audit demonstration")
    p_audit.set_defaults(func=cmd_audit)

    p_train = sub.add_parser("train-models", help="train Layer 02 model checkpoints")
    p_train.set_defaults(func=cmd_train_models)

    p_mdemo = sub.add_parser("model-demo", help="demo multi-source + menopause models")
    p_mdemo.set_defaults(func=cmd_model_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
