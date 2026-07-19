"""CLI: python -m cyclebench.foundation [build|demo|stats]"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_build(_args) -> int:
    from cyclebench.foundation.build import build_foundation
    from cyclebench.foundation.io import DEFAULT_PATH
    b = build_foundation()
    print(
        f"Foundation {b.version}: {len(b.entities)} entities, "
        f"{len(b.associations)} associations, {len(b.evidence)} evidence"
    )
    print(f"→ {DEFAULT_PATH}")
    return 0


def cmd_stats(_args) -> int:
    from cyclebench.foundation.io import load_bundle
    b = load_bundle()
    by_kind = {}
    for e in b.entities:
        by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
    by_rel = {}
    for a in b.associations:
        by_rel[a.relation.value] = by_rel.get(a.relation.value, 0) + 1
    by_ev = {}
    for e in b.evidence:
        by_ev[e.evidence_type.value] = by_ev.get(e.evidence_type.value, 0) + 1
    print(json.dumps({
        "version": b.version,
        "n_entities": len(b.entities),
        "n_associations": len(b.associations),
        "n_evidence": len(b.evidence),
        "entities_by_kind": by_kind,
        "associations_by_relation": by_rel,
        "evidence_by_type": by_ev,
    }, indent=2))
    return 0


def cmd_demo(_args) -> int:
    from cyclebench.foundation.query import assemble_read
    intake = {
        "symptoms": [{"type": "migraine", "severity": "severe"}],
        "symptom_timing": "before_period",
        "contraception_status": "changed",
        "contraception_formulation": "Combined pill",
        "sleep_quality": "bad",
        "age_range": "30-39",
        "last_period_days_ago": 24,
    }
    read = assemble_read(intake)
    print(f"Foundation {read.foundation_version} — {len(read.cards)} cards\n")
    for c in read.cards:
        print(f"## {c.title}")
        print(f"FACT: {c.foundation_fact}")
        for e in c.evidence_summaries:
            print(f"EVIDENCE: {e}")
        if c.personal_pattern:
            print(f"PERSONAL: {c.personal_pattern}")
        print(f"ASK: {c.ask_doctor}")
        print(f"datasets={c.datasets} source={c.source[:60]}…\n")
    if read.model_signals:
        print("MODEL SIGNALS:")
        for m in read.model_signals:
            print(" -", m.get("statement", m)[:160])
    if read.doctor_questions:
        print("\nQUESTIONS:")
        for q in read.doctor_questions:
            print(" •", q)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cyclebench.foundation")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
