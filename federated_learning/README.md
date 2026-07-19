# Federated learning research (optional Track 02)

Hamza’s Personalized Federated Learning (FedPer) experiments from the walkthrough,
path-fixed for this repo.

## What this is

Offline research simulations comparing **Local Only** vs **FedAvg** vs **FedPer (pFL)**
on mcPHASES (and optionally NHANES for multi-site).

| Experiment | Script | Task |
|------------|--------|------|
| multi_site | `federated_personalized_model.py` | Site cohorts (needs `data/raw/P_TST.csv` + demo) |
| patient | `patient_centric_federated_model.py` | Cramps from daily hormones |
| temporal | `temporal_patient_federated_model.py` | Cramps from hormone windows (GRU) |
| multi_symptom | `multi_symptom_federated_model.py` | Cramps from hormones + symptoms (GRU) |

## Requirements

- Local mcPHASES at `data/mcphases/hormones_and_selfreport.csv` (restricted, gitignored)
- Optional: `pip install torch matplotlib` (not required for the Aestra offline demo)

```bash
make install-pfl          # torch + matplotlib into .venv
PFL_ROUNDS=3 make pfl-smoke   # quick multi_symptom run → results/pfl_multi_symptom.json
PFL_ROUNDS=30 make pfl-full   # walkthrough-length run (slow)
```

## Relation to Aestra / CycleBench

- This does **not** replace the deterministic engine, foundation graph, or sklearn models.
- Product AI on `finished` answers different jobs (timeline patterns, phase/PCOS/menopause signals).
- pFL is a **privacy-preserving research story** (cramps prediction across clients without pooling raw rows).

API (optional): `GET /research/pfl/results` reads `results/pfl_*.json` if present.

## Soft-read integration (Aestra)

User-facing name: **Personalized sequence (research)** (not “pFL”).

Research is **opt-in**, not part of the default soft read:

- Under the soft read: **“Research depth · privacy-preserving sequence model”**
- Opens a panel fed by `GET /research/depth` (Local vs FedAvg vs FedPer metrics)
- Soft read stays: patterns → foundation → PCOS/menopause only

Personal estimates still require `intake.sequence_days` (≥5 days) + checkpoint from `make pfl-smoke`.
