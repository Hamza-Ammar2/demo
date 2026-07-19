# Personalized FL (app path on `fed-merge-finished`)

This stack ships a **FedPer-style GRU** that predicts menstrual **phase**
(`Menstrual` / `Follicular` / `Fertility` / `Luteal`) from 21 daily features
(11 ordinal symptoms + 10 wearable/context channels). On the merge branch it is
wired into `POST /models/hormonal-state` **and** the Aestra feeling-off path
(`assemble_read` → Model-pFL → `compose_doctor_followup`). UI/UX remains Aestra
from `finished` — see [FED_MERGE.md](./FED_MERGE.md).

This is **not** the four offline experiments in Hamza’s research walkthrough
(cramps prediction, multi-site NHANES, etc.). Those scripts lived in a separate
`federated_learning/` package — see [PFL_EXPERIMENTS.md](./PFL_EXPERIMENTS.md).

## What you need

| Item | Notes |
|------|--------|
| Python 3.11+ venv | `make install` |
| PyTorch | `make install-pfl` |
| Global checkpoint | `models/global_pfl_model.pt` (committed) or `make pfl-train-global` |
| Feature norms | `models/global_pfl_model.meta.npz` (auto-built on first inference if mcPHASES is present) |
| mcPHASES (restricted) | PhysioNet under `data/mcphases/` — required for global retrain + federated-sync **simulation** |

## Quick reproduce (inference)

```bash
make install
make install-pfl
# Optional: force sklearn instead of pFL
# export CYCLEBENCH_USE_PFL=0

make pfl-smoke          # pad×5 inference against the global checkpoint
# or:
.venv/bin/python -c "
from cyclebench.model.pfl import run_pfl_inference
print(run_pfl_inference({
    'headaches_ord': 2, 'cramps_ord': 4, 'fatigue_ord': 3,
    'sleep_minutes': 380, 'resting_hr': 68, 'steps_sum': 7000,
}))
"
```

Expected fields in the JSON:

- `model`: `Personalized_GRU_FedPer`
- `sequence_padded`: `true` when fewer than 4 prior days were logged
- `honesty_note`: explains pad vs real history

## Local personalization + FedPer simulation

1. Call `/models/hormonal-state` (or `predict_hormonal_state`) several times so rows append to
   `results/local_patient_data.csv` (gitignored).
2. After **≥8** logged days (and ≥5 for windows):

```bash
curl -X POST http://localhost:8000/models/train-local
curl -X POST http://localhost:8000/models/federated-sync
```

Or:

```bash
make pfl-train-local    # needs local CSV
make pfl-sync           # needs mcPHASES peers
```

**Honesty**

- `federated-sync` is a **local simulation**: other mcPHASES subjects act as peers. There is no live clinic network.
- Accuracy before/after sync is returned **only** when a local eval split exists. Fabricated metrics were removed.
- Labels in `local_patient_data.csv` come from the model’s own phase prediction unless you replace them with true labels — treat local fine-tune metrics cautiously.
- Pad×5 (repeat today’s features five times) is a bootstrap for sparse chip/API intake, not true longitudinal history.

## Architecture (short)

```
daily features (21) ──► window W=5 ──► GRUProjection ──► SharedEncoder ──► DecisionHead ──► phase
                              ▲                              ▲
                     local logs / pad×5              FedPer: average peer encoders
```

FedPer keeps the **decision head** local and averages **encoder** weights across peers.

## Environment knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `CYCLEBENCH_USE_PFL` | `1` | Set `0` to use sklearn `hormonal_state_v0.1` only |
| `HF_TOKEN` / `HF_DATASET_REPO` | unset | Optional consented peer sync (see `.env.example`) |

## Makefile targets

| Target | Purpose |
|--------|---------|
| `make install-pfl` | Install `requirements-pfl.txt` (torch) |
| `make pfl-smoke` | One padded inference |
| `make pfl-train-global` | Retrain global checkpoint from mcPHASES |
| `make pfl-train-local` | Fine-tune from `results/local_patient_data.csv` |
| `make pfl-sync` | One FedPer simulation round |

## Relation to sklearn hormonal-state

The sklearn joblib model remains available as fallback (missing torch / checkpoint / errors)
and when `CYCLEBENCH_USE_PFL=0`. On this branch, successful pFL inference **replaces** the
sklearn phase for `/models/hormonal-state`.
