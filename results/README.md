# Results directory

Aggregate, redistributable metrics and evaluation outputs. **No participant-level rows.**

| File | What it is | Synthetic / real |
|------|------------|------------------|
| `benchmark_results.json` | CycleBench-Bench Path A vs B | Cases are **synthetic** (see BENCHMARK.md) |
| `model_hormonal_state.json` | Hormonal-state metrics + importances | Trained on **mcPHASES** (restricted; not shipped) |
| `model_menopause_stage.json` | Menopause-stage metrics | Check `data_source` — often **`synthetic_swan_like`** |
| `model_pcos_risk.json` | PCOS-risk metrics | **Kaggle PCOS** (raw not shipped) |
| `model_train_summary.json` | Train rollup | Includes menopause `data_source` |
| `mcphases_validation.json` | Aggregate validation | Real mcPHASES aggregates only |
| `reference_stats.json` | App grounding aggregates | Mix of open + local aggregates |

Regenerate:

```bash
make benchmark
make train-models && make train-tasks
make mcphases    # needs restricted data
make reference
```
