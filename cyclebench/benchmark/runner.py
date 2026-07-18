"""CycleBench-Bench v0.1 runner.

Evaluates Path A (naive summarizer) vs Path B (CycleBench engine) on the synthetic
cases and writes real metrics to /results. Deterministic; no network, no API key.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cyclebench.benchmark.baselines import engine_analysis, naive_summary
from cyclebench.benchmark.cases import BENCH_CASES
from cyclebench.safety import find_violations

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BENCH_VERSION = "CycleBench-Bench v0.1"


def _reading_seconds(text_words: int) -> float:
    return round(text_words / 200.0 * 60.0, 1)  # 200 wpm


def _brief_words(brief) -> int:
    parts = ([brief.opening_statement] + brief.strongest_findings + brief.unresolved_questions
             + brief.established + brief.possible + brief.not_established + brief.missing)
    return sum(len(p.split()) for p in parts)


def evaluate() -> dict:
    per_case = []
    for bc in BENCH_CASES:
        a = naive_summary(bc.case)
        b = engine_analysis(bc.case)

        a_safety = sum(len(find_violations(s)) for s in a["statements"])
        b_safety = sum(len(find_violations(s)) for s in b["statements"])

        # unsupported claims = asserted pattern statements with no provenance
        a_unsupported = len(a["statements"]) if not a["provenance_present"] else 0
        b_unsupported = 0 if b["provenance_present"] else len(b["statements"])

        narrative_words = len(bc.narrative.split())
        brief_words = _brief_words(b["brief"])

        per_case.append({
            "case_id": bc.case_id,
            "category": bc.category,
            "expected_pattern": bc.expect_cyclical_pattern,
            "A_pattern": a["claims_pattern"],
            "B_pattern": b["claims_pattern"],
            "A_correct": a["claims_pattern"] == bc.expect_cyclical_pattern,
            "B_correct": b["claims_pattern"] == bc.expect_cyclical_pattern,
            "expected_confounders": sorted(bc.expect_confounders),
            "B_confounders": sorted(b["detected_confounders"]),
            "expected_missing": sorted(bc.expect_missing_fields),
            "B_missing": sorted(b["detected_missing"]),
            "expected_change": bc.expect_change_after_event,
            "B_change": b["detected_change_after_event"],
            "A_safety_violations": a_safety,
            "B_safety_violations": b_safety,
            "A_unsupported_claims": a_unsupported,
            "B_unsupported_claims": b_unsupported,
            "B_provenance_ok": b["provenance_present"],
            "narrative_words": narrative_words,
            "brief_reading_seconds": _reading_seconds(brief_words),
        })

    n = len(per_case)

    def acc(key):
        return round(sum(1 for r in per_case if r[key]) / n, 3)

    neg = [r for r in per_case if not r["expected_pattern"]]

    def false_pattern(col):
        return round(sum(1 for r in neg if r[col]) / len(neg), 3) if neg else None

    conf_cases = [(bc, r) for bc, r in zip(BENCH_CASES, per_case) if bc.expect_confounders]
    conf_recall = None
    if conf_cases:
        hits = sum(
            len(set(r["B_confounders"]) & bc.expect_confounders) / len(bc.expect_confounders)
            for bc, r in conf_cases
        )
        conf_recall = round(hits / len(conf_cases), 3)

    miss_cases = [(bc, r) for bc, r in zip(BENCH_CASES, per_case) if bc.expect_missing_fields]
    miss_recall = None
    if miss_cases:
        hits = sum(
            len(set(r["B_missing"]) & bc.expect_missing_fields) / len(bc.expect_missing_fields)
            for bc, r in miss_cases
        )
        miss_recall = round(hits / len(miss_cases), 3)

    change_cases = [r for r in per_case if r["expected_change"]]
    change_recall = (
        round(sum(1 for r in change_cases if r["B_change"]) / len(change_cases), 3)
        if change_cases else None
    )

    summary = {
        "benchmark": BENCH_VERSION,
        "n_cases": n,
        "categories": {c: sum(1 for r in per_case if r["category"] == c)
                       for c in sorted({r["category"] for r in per_case})},
        "pattern_detection_accuracy": {"path_A_naive": acc("A_correct"),
                                       "path_B_cyclebench": acc("B_correct")},
        "false_pattern_rate": {"path_A_naive": false_pattern("A_pattern"),
                               "path_B_cyclebench": false_pattern("B_pattern")},
        "confounder_recall_B": conf_recall,
        "missing_info_recall_B": miss_recall,
        "change_after_event_recall_B": change_recall,
        "provenance_coverage_B": acc("B_provenance_ok"),
        "unsupported_claim_count": {"path_A_naive": sum(r["A_unsupported_claims"] for r in per_case),
                                    "path_B_cyclebench": sum(r["B_unsupported_claims"] for r in per_case)},
        "safety_violation_count": {"path_A_naive": sum(r["A_safety_violations"] for r in per_case),
                                   "path_B_cyclebench": sum(r["B_safety_violations"] for r in per_case)},
        "mean_narrative_words": round(sum(r["narrative_words"] for r in per_case) / n, 1),
        "mean_brief_reading_seconds": round(sum(r["brief_reading_seconds"] for r in per_case) / n, 1),
    }
    return {"summary": summary, "per_case": per_case}


def main() -> int:
    out = evaluate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark_results.json").write_text(json.dumps(out, indent=2))

    with open(RESULTS / "benchmark_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out["per_case"][0].keys()))
        writer.writeheader()
        writer.writerows(out["per_case"])

    s = out["summary"]
    print(json.dumps(s, indent=2))
    print("\nPer-case pattern calls (expected | A_naive | B_cyclebench):")
    for r in out["per_case"]:
        print(f"  {r['case_id']} [{r['category']:>12}]  "
              f"{str(r['expected_pattern']):>5} | A={str(r['A_pattern']):>5} | B={str(r['B_pattern']):>5}"
              f"  {'A✗' if not r['A_correct'] else '  '} {'B✗' if not r['B_correct'] else '  '}")
    print(f"\nResults written to results/benchmark_results.json and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
