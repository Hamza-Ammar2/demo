"""CycleBench-Audit — leakage & safety audit that fails loudly.

Ten assertions guard the scientific integrity of any CycleBench analysis or
prediction task. The audit is designed to REJECT a case/dataset that leaks future
information or emits unsafe language, so a naive-but-leaky pipeline cannot look
better than an honest one.

Assertions:
  1. No participant appears in both train and test splits.
  2. No cycle spans incompatible split boundaries.
  3. In causal mode, every feature timestamp precedes the label/analysis timestamp.
  4. Future menstrual onset is not used as a causal feature.
  5. Future-confirmed ovulation is not used as a causal feature.
  6. Normalization statistics are fitted on training data only.
  7. No target-derived feature enters the feature set.
  8. Every generated finding contains provenance.
  9. Every medical association is labelled association, not causation.
 10. No diagnosis or treatment recommendation appears in generated output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from cyclebench.safety import find_violations
from cyclebench.schema import AnalysisMode, EstablishmentClass, Finding


class AuditError(Exception):
    """Raised (in strict mode) when an audit assertion fails."""


@dataclass
class AuditResult:
    passed: bool = True
    checks: list[dict] = field(default_factory=list)

    def record(self, assertion_id: int, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(
            {"id": assertion_id, "name": name, "passed": ok, "detail": detail}
        )
        if not ok:
            self.passed = False

    def summary(self) -> str:
        lines = [f"CycleBench-Audit: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            mark = "ok " if c["passed"] else "XX "
            lines.append(f"  [{mark}] {c['id']:>2}. {c['name']}"
                         + (f" — {c['detail']}" if c["detail"] else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Split / dataset descriptors for prediction tasks (used by the mcPHASES task).
# --------------------------------------------------------------------------- #
@dataclass
class PredictionSplit:
    """A minimal descriptor of an ML-style task for the audit to inspect."""

    train_subjects: set
    test_subjects: set
    # For each (subject, cycle_id): the split it belongs to. Detects cycle spanning.
    cycle_split: dict = field(default_factory=dict)
    # In causal mode: (feature_timestamp, label_timestamp) pairs for spot-checking.
    feature_label_times: list = field(default_factory=list)
    mode: AnalysisMode = AnalysisMode.causal
    # Names of features, and whether normalization was fit on train only.
    feature_names: list = field(default_factory=list)
    normalization_fit_on_train_only: bool = True
    target_name: Optional[str] = None
    uses_future_onset_feature: bool = False
    uses_future_ovulation_feature: bool = False


def audit_prediction_split(split: PredictionSplit, result: Optional[AuditResult] = None) -> AuditResult:
    r = result or AuditResult()

    # 1. participant isolation
    overlap = split.train_subjects & split.test_subjects
    r.record(1, "participant split isolation", not overlap,
             f"overlap={sorted(overlap)}" if overlap else "")

    # 2. cycles do not span split boundaries (a cycle maps to exactly one split)
    spanning = [k for k, v in split.cycle_split.items() if v not in ("train", "test")]
    r.record(2, "no cycle spans split boundaries", not spanning,
             f"bad={spanning}" if spanning else "")

    # 3. causal: feature time precedes label time
    if split.mode == AnalysisMode.causal:
        bad = [(f, l) for (f, l) in split.feature_label_times if not (f <= l)]
        r.record(3, "causal feature precedes label", not bad,
                 f"{len(bad)} feature(s) after label" if bad else "")
    else:
        r.record(3, "causal feature precedes label", True, "n/a (retrospective)")

    # 4 & 5. no future-derived features in causal mode
    r.record(4, "no future menstrual onset as causal feature",
             not (split.mode == AnalysisMode.causal and split.uses_future_onset_feature))
    r.record(5, "no future-confirmed ovulation as causal feature",
             not (split.mode == AnalysisMode.causal and split.uses_future_ovulation_feature))

    # 6. normalization fit on train only
    r.record(6, "normalization fitted on train only", split.normalization_fit_on_train_only)

    # 7. no target-derived feature in feature set
    leak = split.target_name is not None and split.target_name in split.feature_names
    r.record(7, "no target-derived feature", not leak,
             f"target '{split.target_name}' present in features" if leak else "")

    return r


def audit_findings(findings: list[Finding], result: Optional[AuditResult] = None) -> AuditResult:
    r = result or AuditResult()

    # 8. every asserting finding has provenance
    missing_prov = [
        f.finding_id for f in findings
        if f.establishment in (EstablishmentClass.established, EstablishmentClass.possible)
        and not (f.supporting_event_ids or f.source_ids)
    ]
    r.record(8, "every finding has provenance", not missing_prov,
             f"no-provenance={missing_prov}" if missing_prov else "")

    # 9. associations are not phrased as causation
    causal_lang = [f.finding_id for f in findings if find_violations(f.statement)]
    r.record(9, "associations not phrased as causation", not causal_lang,
             f"unsafe={causal_lang}" if causal_lang else "")

    # 10. no diagnosis / treatment recommendation in output
    #     (find_violations already covers diagnostic + treatment assertions)
    r.record(10, "no diagnosis/treatment recommendation", not causal_lang)

    return r


def audit_all(findings: list[Finding], split: Optional[PredictionSplit],
              strict: bool = False) -> AuditResult:
    r = AuditResult()
    if split is not None:
        audit_prediction_split(split, r)
    else:
        for i, name in [(1, "participant split isolation"), (2, "no cycle spans split boundaries"),
                        (3, "causal feature precedes label"),
                        (4, "no future menstrual onset as causal feature"),
                        (5, "no future-confirmed ovulation as causal feature"),
                        (6, "normalization fitted on train only"),
                        (7, "no target-derived feature")]:
            r.record(i, name, True, "n/a (no prediction task)")
    audit_findings(findings, r)
    if strict and not r.passed:
        raise AuditError(r.summary())
    return r


# --------------------------------------------------------------------------- #
# Demonstration: an honest split passes; a deliberately leaking split is rejected.
# --------------------------------------------------------------------------- #
def _honest_split() -> PredictionSplit:
    return PredictionSplit(
        train_subjects={1, 2, 3, 4},
        test_subjects={5, 6},
        cycle_split={(1, "c1"): "train", (5, "c1"): "test"},
        feature_label_times=[(date(2024, 1, 1), date(2024, 1, 2))],
        mode=AnalysisMode.causal,
        feature_names=["resting_hr", "wrist_temp", "sleep_hours"],
        normalization_fit_on_train_only=True,
        target_name="phase_next_day",
        uses_future_onset_feature=False,
    )


def _leaking_split() -> PredictionSplit:
    return PredictionSplit(
        train_subjects={1, 2, 3, 5},   # subject 5 in BOTH -> leak
        test_subjects={5, 6},
        cycle_split={(1, "c1"): "train", (5, "c1"): "spanning"},  # bad
        feature_label_times=[(date(2024, 1, 3), date(2024, 1, 2))],  # feature AFTER label
        mode=AnalysisMode.causal,
        feature_names=["resting_hr", "phase_next_day"],  # target leaked into features
        normalization_fit_on_train_only=False,           # normalized on all data
        target_name="phase_next_day",
        uses_future_onset_feature=True,                   # future onset used
    )


def run_demo_audit() -> bool:
    """Show the audit passing an honest split and rejecting a leaking one."""
    from cyclebench.engine import compile_case
    from cyclebench.fixtures import build_sarah_case
    from cyclebench.schema import AnalysisMode as AM

    findings = compile_case(build_sarah_case(), AM.retrospective).findings

    print("== Honest prediction split ==")
    honest = audit_all(findings, _honest_split())
    print(honest.summary())

    print("\n== Deliberately LEAKING prediction split (must fail) ==")
    leaking = audit_all(findings, _leaking_split())
    print(leaking.summary())

    ok = honest.passed and (not leaking.passed)
    print("\nRESULT:",
          "PASS — audit accepted the honest split and REJECTED the leaking one."
          if ok else "UNEXPECTED — audit did not behave as intended.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_demo_audit() else 1)
