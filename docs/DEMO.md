# Demo Script (3 minutes)

**Setup:** `make install` once, then `make api` → open http://localhost:8000.
Everything runs offline; no API key required.

## 0:00 — The problem
"Sarah has months of migraines, period changes, a pill switch, and bad sleep. She gets ten
minutes with her doctor. Today she recites fragments from memory."

## 0:30 — Compile the timeline (click **Load Sarah demo**)
The Doctor Brief appears: opening statement, three strongest findings, three unresolved
questions, and four buckets (Established / Possible / Not established / Missing).
"Every sentence here is a calculation, not a guess."

## 1:15 — Provenance & confounders
Scroll to **findings with provenance**: the cyclical-migraine finding links to the exact
supporting events, and flags reduced sleep as a confounder. "It won't let itself claim a
pattern without showing its evidence — and it volunteers the alternative explanation."

## 1:45 — Honesty across modes (click **Mode: retrospective → causal**)
Retrospective finds **4 of 5** migraines in the luteal window; causal reports only **3 of 5**,
because the first episode can't be phase-assigned without peeking at a future period.
"Most tools would quietly use the future. Ours tells you when it can't know yet."

## 2:15 — The audit (click **Run leakage audit**)
The honest split passes all 10 assertions; the deliberately leaking split is **rejected**.
"This is the trick that makes naive models look smart. We catch it."

## 2:40 — The benchmark (click **View the benchmark**)
Naive summarizer: 83% false-pattern rate, unsupported causal claims. CycleBench: 0%
false-pattern rate, 100% provenance, 0 safety violations. "Reproducible with `make benchmark`."

## 2:55 — The open contribution
"Schema, engine, audit, benchmark, and a harmonized open NHANES dataset — MIT/CC-BY, ready to
reuse. We grounded it on real mcPHASES data without ever redistributing a single restricted row."
