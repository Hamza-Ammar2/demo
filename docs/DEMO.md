# Demo Script (~3 minutes)

**Setup:** `make install` once, then `make api` → open http://127.0.0.1:8000.
Everything runs offline; no API key required. Full checklist: `docs/REPRODUCIBILITY.md`.

## 0:00 — The problem
"Sarah has months of migraines, period changes, a pill switch, and bad sleep. She gets ten
minutes with her doctor. Today she recites fragments from memory."

## 0:25 — The product (Aestra)
Open **I've been feeling off**. Show chip-based intake → noted chips → soft read takeover.
"This is the application layer. The medicine isn't invented by a chatbot."

## 1:00 — The infrastructure (CycleBench)
Name the open blocks: schema, deterministic engine, leakage audit, benchmark, foundation
graph, specialist models. "Judges: this is Track 01 + 02; Aestra is Track 03."

## 1:40 — Honesty
- LLM never computes findings.
- mcPHASES: aggregates only, raw never shipped.
- Menopause-stage: disclose `synthetic_swan_like` if that's what `results/model_menopause_stage.json` says.
- Soft read: associations for a doctor conversation — not a diagnosis.

## 2:10 — Audit + benchmark (CLI or UI if available)
`make audit` — leaking split rejected.
`make benchmark` — naive vs engine metrics in `results/`.

## 2:45 — Close
"Reusable under MIT / CC-BY. Clone, `make install`, `make test`, `make api` — no keys required."
