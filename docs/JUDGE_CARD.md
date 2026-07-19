# Judge card — honest numbers & track map

One page for reviewers. Full methodology: `docs/REPRODUCIBILITY.md`, `docs/MODEL_CARD.md`, `docs/BENCHMARK.md`.

## What this is

**CycleBench** = open scientific stack (schema, engine, audit, benchmark, foundation, specialist models).  
**Aestra** = appointment-prep app on top (chip intake → soft doctor brief).  
Not a diagnostic device. LLM only phrases language; it never invents findings.

## Track map (Hack-Nation)

| Track | What we leave behind | Strength |
|-------|----------------------|----------|
| **01 Data** | Open NHANES export + datasheet; dataset registry (supports / must-not); foundation graph; CycleBench-Bench v0.1; aggregate mcPHASES validation (raw not shipped) | Adequate — strong honesty; no redistributable multimodal longitudinal pack |
| **02 Models** | PCOS-risk, menopause-stage, hormonal-state (sklearn), personalized FedPer GRU phase; participant/stratified splits; leakage audit; explainability on sklearn | Adequate — PCOS strong; phase weak; menopause checkpoint synthetic until SWAN retrain |
| **03 App** | Aestra feeling-off soft read; Model-pFL → doctor follow-up; optional consented research logs | Strong demo — regulator-light contribution, not full EHR pipeline |

## Open vs restricted vs synthetic

| Asset | Redistributable? | Notes |
|-------|------------------|-------|
| `data/nhanes_harmonized/` | **Yes** (CC-BY-4.0) | Cross-sectional hormone refs |
| `data/foundation/` | Yes (code+seed) | Medical foundation graph |
| CycleBench-Bench cases | Yes (synthetic) | Co-designed to test discrimination — see honesty note in results |
| mcPHASES raw | **No** | PhysioNet DUA; aggregates only in `results/mcphases_validation.json` |
| SWAN raw | **No** | ICPSR; see `docs/SWAN_ACCESS.md` |
| PCOS Kaggle CSV | **No** | Retrain with your download |
| Menopause checkpoint metrics | **Synthetic label** | `data_source: synthetic_swan_like` in shipped JSON |

## Model metrics (from committed `results/model_*.json`)

| Model | Balanced acc | Acc | Macro F1 | `data_source` | Cite as |
|-------|--------------|-----|----------|---------------|---------|
| Hormonal-state (sklearn) | **0.28** | 0.31 | 0.27 | `mcphases` | Barely above chance (0.25); honest |
| Menopause-stage | **0.99** | 0.99 | 0.99 | `synthetic_swan_like` | **Illustrative only** until real SWAN retrain |
| PCOS-risk | **0.86** | 0.90 | 0.88 | `kaggle_pcos` | Strong association signal; not a diagnosis |
| pFL GRU phase | — | ~**0.51** held-out W=5* | ~0.50* | mcPHASES (local eval) | Better than sklearn with real history; pad×5 ≈ chance |

\*pFL holdout numbers from local evaluation on this machine (participant-style); not a committed JSON yet. App sparse intake often uses pad×5 → ~chance. See `docs/PFL.md`.

## Benchmark (CycleBench-Bench v0.1)

See `results/benchmark_results.json`: Path B (engine) vs naive summarizer. Cases are synthetic and co-designed with engine thresholds — discrimination / leakage / safety test, **not** clinical generalization.

## Offline reproduce (no API keys)

```bash
make install
make test
make demo
make audit
make benchmark
make api          # http://127.0.0.1:8000  Aestra UI
```

Optional pFL: `make install-pfl && make pfl-smoke`.

## Contribution path (Track 03, opt-in)

1. Soft-read checkbox (default **off**) → `POST /models/consent`
2. Local file `results/user_consent.txt`
3. Optional anonymized phase CSV upload if `HF_TOKEN` + `HF_DATASET_REPO` set
4. Consented sessions also append to `data/contributions/sessions.jsonl` (gitignored)

## Demo script (≈90s)

1. Open Aestra → “I've been feeling off”
2. Chips: cramps + migraine, timing around period, last period days ago
3. Soft read: **model signals first** (phase / optional PCOS–meno) → medical foundation → ask-doctor
4. Say out loud: “Model-pFL estimates phase; Model-agent only phrases the doctor follow-up — it cannot invent a phase.”
5. Disclaimer: associations for conversation, not diagnosis.

## Video / narrative brief

Give ChatGPT / yourself: root **`logic.md`**.
