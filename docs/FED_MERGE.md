# fed-merge-finished — what was merged

Branch base: **`finished`** (Aestra UI/UX, foundation, modular specialist models).  
Brought in from **`fed`**: personalized FedPer GRU phase model (`cyclebench/model/pfl.py`,
`models/global_pfl_model.pt`, train-local / federated-sync APIs).

## Product wiring (Hamza’s core logic)

```
User intake (Aestra UI)
        │
        ▼
 Model-pFL  ──► phase estimate (+ optional local fine-tune / FedPer sim)
        │
        ▼
 Model-agent / LLM  ──► doctor follow-ups, how-to-explain, questions
        │                 (never invents a new phase; safety-guarded)
        ▼
 Soft read (finished UX) — model chips + ask-doctor list
```

| Layer | Where |
|-------|--------|
| Model-pFL | `cyclebench/model/pfl.py` → `predict_hormonal_state` / foundation `_runtime_model_signals` |
| Intake → features | `cyclebench.intake.intake_to_pfl_features` |
| Model-agent | `cyclebench.llm.compose_doctor_followup` (offline template if no API key) |
| UI | **unchanged Aestra** `web/` from `finished` (+ phase chip label) |

## Datasets — kept modular (not one fused net)

We did **not** delete the finished scientific stack. Specialists stay separate:

| Dataset | Role on this branch |
|---------|---------------------|
| **mcPHASES** | Phase pFL + sklearn hormonal-state (restricted; local only) |
| **PCOS Kaggle** | Separate PCOS-risk task when irregular+cluster tags match |
| **SWAN** | Separate menopause-stage model when midlife-relevant |
| **NHANES** | Reference / foundation evidence (open export) |

What we **did not** merge in:

- Hamza’s offline `federated_learning/` Exp 1–4 (cramps / multi-site) — see `docs/PFL_EXPERIMENTS.md`
- Any “one model eats all datasets” fusion Hamza criticized as fake when done poorly

If disk or licensing pressure forces cuts later, prefer dropping **research-only** packages
before product specialists. Do not remove mcPHASES if you want pFL to train/sync.

## Reproduce

```bash
make install
make install-pfl
make foundation      # picks up assoc.model_phase
make pfl-smoke
make api             # Aestra UI at :8000
```

Feeling-off path returns `phase_model`, `doctor_followup`, and updated ask-doctor questions.

## Hamza `fed` upgrades ported (commit `database setup`)

| Upgrade | How we took it |
|---------|----------------|
| `global_scaler.pt` | Shared train/infer/sync scaling |
| Consent + anonymous client id | `/models/consent`, Aestra checkbox + analyse-full step |
| Hugging Face peer dataset | `/models/huggingface-sync` + peer backend when `HF_*` set |
| Offline peers | Still works via `results/hf_export/data` or `data/mcphases/` |
| Fake sync accuracies (`0.65`/`0.825`) | **Not** ported |
| Committing redistributed mcPHASES CSVs | **Not** ported (gitignored; DUA risk) |
