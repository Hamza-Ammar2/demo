# Model Card — CycleBench Layer 02 (v0.1)

## Models
1. **hormonal_state_v0.1** — multi-source hormonal-state (cycle phase) classifier  
2. **menopause_stage_v0.1** — menopause-stage category classifier  
3. **pcos_risk_v0.1** — PCOS-risk classifier (model factory / Kaggle cohort)

## Intended use
Research / appointment-preparation context signals. Outputs are probabilities with
feature attributions, wrapped as CycleBench `Finding`s (`establishment=possible`).
**Not** a medical device. **Not** a diagnosis.

## Training data
| Model | Data | Notes |
|---|---|---|
| hormonal_state | mcPHASES (PhysioNet, restricted) | Wearables + sleep + symptoms + CGM → Mira phase. Hormone metabolites excluded to avoid label leakage. Participant-held-out split. |
| menopause_stage | SWAN ICPSR when `data/swan/swan_harmonized.csv` present; else synthetic SWAN-like cohort | Stage category from hormones + vasomotor/sleep/cycle features + age. Check `data_source` in metrics JSON. |
| pcos_risk | Kaggle PCOS (Kerala clinic cohort) | See `results/model_pcos_risk.json` for provenance. Raw CSV gitignored. |

## Metrics
See `results/model_hormonal_state.json`, `results/model_menopause_stage.json`, and
`results/model_pcos_risk.json` after training.

## Explainability
Gradient-boosting feature importances + per-prediction contribution scores
(|scaled feature| × importance). Surfaced in API + CLI.

## Ethical / safety
- Association language only (safety guard applies when findings enter briefs).
- Restricted mcPHASES rows never redistributed; checkpoints may be shared.
- Synthetic menopause fallback must be disclosed whenever used.

## How to reproduce
```bash
make train-models
make model-demo
```
