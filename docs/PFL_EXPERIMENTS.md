# Offline pFL experiments (research walkthrough)

Hamza’s research walkthrough ([Desktop/walkthrough.md](file:///Users/lukepons/Desktop/walkthrough.md))
evaluates **four** Personalized Federated Learning setups under **FedPer**, mostly for
**cramps** (and one multi-site NHANES + mcPHASES cohort task). Those pipelines lived in a
separate `federated_learning/` directory and are **not** the app code on this `fed` branch.

The `fed` branch app path is documented in [PFL.md](./PFL.md): phase classification with a
21-feature GRU, pad×5 bootstrap, and a local FedPer sync simulation.

## Experiment summary (from the research walkthrough)

Numbers below are copied from that walkthrough for orientation. Re-running them requires
the original `federated_learning/` scripts + local data paths — they are not Makefile
targets on this branch.

### Experiment 1 — Multi-cohort / site

Three simulated clinic clients (balanced NHANES, male-skewed NHANES, mcPHASES cyclic).
pFL often beat FedAvg under heterogeneity (see walkthrough tables).

### Experiment 2 — Patient-centric, non-temporal

Each mcPHASES subject = client. Task: daily cramps (imbalanced; `pos_weight` in BCE).
Average across 19 patients: pFL accuracy ≈ **0.74** (local ≈ 0.72, FedAvg ≈ 0.71).

### Experiment 3 — Temporal GRU, hormones only

Window **W = 5**. Twelve clients with enough sequences. Average pFL accuracy ≈ **0.89**,
F1 ≈ **0.61**.

### Experiment 4 — Multi-symptom temporal fusion

10 channels (3 hormones + 7 symptoms). Average pFL accuracy ≈ **0.91**, F1 ≈ **0.70**,
specificity ≈ **0.95**.

## How this relates to the app

| | Research experiments | `fed` app (`docs/PFL.md`) |
|--|----------------------|---------------------------|
| Task | Mostly cramps (binary / ordinal) | 4-class **phase** |
| Package | `federated_learning/*.py` | `cyclebench/model/pfl.py` |
| Network | Multi-round training loops | One-shot local sync simulation |
| Chip intake | Not the focus | Pad×5 for sparse days |

If you restore the research package into this repo, document install + exact commands here
and keep results under `results/pfl_experiments/` (gitignored raw data still applies).
