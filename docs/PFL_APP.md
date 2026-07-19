# App walkthrough: pFL on Aestra (`fed-merge-finished`)

Canonical technical doc: [PFL.md](./PFL.md). Merge notes: [FED_MERGE.md](./FED_MERGE.md).

## Branch

- **Branch**: `fed-merge-finished` (Aestra UI from `finished` + Hamza pFL upgrades)
- **Core module**: `cyclebench/model/pfl.py`
- **Checkpoint**: `models/global_pfl_model.pt` (+ `models/global_scaler.pt`)

## Integrated pieces

1. **GRU client model** — 21 inputs → 4 phases; FedPer splits shared encoder vs local head.
2. **Local logger** — appends to `results/local_patient_data.csv` (gitignored).
3. **Inference hook** — `predict_hormonal_state` prefers pFL when `CYCLEBENCH_USE_PFL=1` (default).
4. **Feeling-off** — foundation `model_signals` + `compose_doctor_followup` (LLM/agent layer).
5. **API** — `POST /models/train-local`, `/models/federated-sync`, `/models/consent`, `/models/huggingface-sync`.

## Verify

```bash
make install && make install-pfl
make pfl-smoke
make api
```

Look for `"model": "Personalized_GRU_FedPer"` and an honest `"sequence_padded"` flag.
Soft-read UI shows **Model signals** above Medical foundation.
