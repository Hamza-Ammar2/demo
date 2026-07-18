# Decisions & Assumptions Log

Key judgment calls made during the build, with rationale.

1. **LLM never computes findings.** All analytics are deterministic; the LLM is optional and
   confined to extraction + phrasing. Rationale: auditability, reproducibility, safety.

2. **Two explicit analysis modes (retrospective vs causal).** Mode is mandatory on
   `CycleContext`; causal forbids future data at the schema level. Rationale: the single most
   important safeguard against future-information leakage in cycle claims.

3. **Provenance-or-silence enforced by schema validation.** An asserting finding without
   evidence cannot be constructed. Rationale: anti-hallucination guarantee.

4. **Safety enforced in code, negation-aware.** We flag affirmative diagnostic/causal/
   treatment claims but allow negated/hedged phrasing. Rationale: prompt-only guardrails are
   not verifiable; also avoids false positives on disclaimers ("does not diagnose").

5. **Do NOT predict hormone levels or menopause.** The available data (consumer urine proxy,
   n=42; cross-sectional serum) cannot support it. Rationale: scientific honesty > flashy demo.
   We detect and audit *patterns* instead.

6. **mcPHASES: aggregate-only, never redistributed.** Restricted license. We publish only
   summary statistics and adapter code. Rationale: legal + ethical compliance.

7. **NHANES: publish a harmonized CC-BY export.** Public domain, so we can maximize open-science
   value with reference ranges.

8. **Static frontend served by FastAPI instead of Next.js.** Local Node is v14 (too old).
   Rationale: a zero-build static page is demo-proof and removes a failure mode; the schema/
   engine (the reusable asset) is unaffected.

9. **Benchmark is synthetic + documented as such.** Path B accuracy is high because cases and
   thresholds are co-designed; value is methodology, negative/misleading coverage, and safety —
   not a leaderboard number. Real-data grounding is reported separately (mcPHASES).

10. **Benchmark case fixes for rigor:** irregular-cycle case expects *no* confident pattern;
    change-after-medication case spreads after-episodes across phases so the change signal is
    isolated from spurious cyclical clustering.

11. **In-memory, non-persistent API store.** No health data is written to disk by the API.
    Rationale: privacy by default for a prototype.
