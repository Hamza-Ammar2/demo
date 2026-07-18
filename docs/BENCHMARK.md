# CycleBench-Bench v0.1 — Benchmark Methodology

## Question
Can a system turn fragmented longitudinal input into an **accurate and safe** doctor brief —
including correctly reporting **no meaningful pattern** when none exists?

## Cases
10 documented, **synthetic** cases (`cyclebench/benchmark/cases.py`), labeled synthetic:
- 4 positive (cyclical clustering / change-after-event)
- 2 negative (symptoms spread across the cycle)
- 1 misleading (looks cyclical, but every episode overlaps poor sleep)
- 1 irregular (high cycle-length variability → must withhold a confident claim)
- 2 insufficient (too little data to align)

Each case carries ground-truth structured events plus expected outcomes
(`expect_cyclical_pattern`, `expect_confounders`, `expect_missing_fields`,
`expect_change_after_event`).

## Two evaluation paths
- **Path A — naive summarizer** (`baselines.naive_summary`): claims a hormonal pattern
  whenever a symptom repeats ≥3×, with no cycle alignment, no provenance, and causal phrasing.
  Documented as an **illustrative** stand-in for an unconstrained generic summary.
- **Path B — CycleBench engine** (`baselines.engine_analysis`): the deterministic pipeline.
  A pattern is "claimed" only if relative frequency ≥1.5, ≥3 aligned episodes, and confidence
  ≥ medium.

## Metrics
pattern-detection accuracy, false-pattern rate, confounder recall, missing-info recall,
change-after-event recall, provenance coverage, unsupported-claim count, safety-violation
count, brief reading time. Deterministic; fixed by construction; run with `make benchmark`.

## Results (reproducible; see `results/benchmark_results.json`)

| Metric | Naive (A) | CycleBench (B) |
|---|---|---|
| pattern-detection accuracy | 0.50 | **1.00** |
| false-pattern rate | 0.833 | **0.00** |
| unsupported claims | 9 | **0** |
| safety violations | 18 | **0** |
| confounder recall | — | 1.00 |
| missing-info recall | — | 1.00 |
| provenance coverage | — | 1.00 |

## Honesty / validity notes
- These are **constructed** cases that validate the engine's *discrimination and safety*, not
  an independent held-out test set; Path B's perfect accuracy reflects that the cases and the
  engine's decision thresholds are co-designed and transparent. The scientific value is the
  documented methodology, the negative/misleading/irregular coverage, and the safety/provenance
  guarantees — not a leaderboard number.
- Path A is deliberately simple; it is a floor, not a state-of-the-art LLM baseline.
- Real-data grounding lives separately in `results/mcphases_validation.json` (see DATASETS).
