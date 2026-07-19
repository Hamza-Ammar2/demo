# Models directory

Trained checkpoints for CycleBench Layer 02 / model factory.

| File | Task | Read metrics |
|------|------|----------------|
| `hormonal_state_v0.1.joblib` | Cycle phase (sklearn fallback) | `results/model_hormonal_state.json` |
| `menopause_stage_v0.1.joblib` | Menopause stage category | `results/model_menopause_stage.json` |
| `pcos_risk_v0.1.joblib` | PCOS risk | `results/model_pcos_risk.json` |
| `global_pfl_model.pt` | Personalized FedPer GRU phase | see `docs/PFL.md` / `docs/JUDGE_CARD.md` |
| `global_scaler.pt` | Feature mean/std for pFL | paired with `global_pfl_model.pt` |

Sklearn checkpoints have sibling `*.meta.json`. Local personalization weights
(`local_patient_model.pt`) are gitignored.

## Honesty rules before you cite numbers

1. Open the matching `results/model_*.json` and read **`data_source`** (and any disclaimer).
2. If menopause says `synthetic_swan_like`, treat balanced accuracy as **illustrative**.
3. mcPHASES raw rows are **not** in this repo; hormonal-state was trained where that data
   was available locally under a PhysioNet DUA.
4. PCOS raw Kaggle CSV is **not** redistributed; re-download to retrain (`make train-tasks`).

Reproduce:

```bash
make train-models   # hormonal + menopause
make train-tasks    # PCOS (needs data/kaggle/…)
make model-demo
```

See `docs/MODEL_CARD.md` and `docs/REPRODUCIBILITY.md`.
