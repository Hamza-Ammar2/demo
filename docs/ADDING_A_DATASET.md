# Adding a dataset to the CycleBench foundation

Goal: strengthen the medical foundation with evidence — without rewriting the chat,
without letting an LLM invent medical edges, and without leaking restricted rows.

## 10 steps

1. **Get the data locally** under `data/` (gitignored if restricted/licensed).
2. **Register it** in `cyclebench/adapters/registry.py` with clear `supports` / `does_not_support`.
3. **Map columns → Entity IDs** in `cyclebench/foundation/seed.py` (add entities only if missing).
4. **Decide which Associations** the dataset can strengthen (prefer existing edges).
   - Do **not** create a new clinical association from a weak correlation alone.
5. **Write an evidence adapter** in `cyclebench/foundation/evidence.py`:
   - emit `Evidence` records pointing at `association_id`s
   - include metrics + safe `summary_sentence` + license note
   - run `assert_safe` on every sentence
6. **Call your adapter** from `attach_all_evidence()`.
7. **Rebuild**: `make foundation` (and `make reference` / `make train-tasks` if needed).
8. **Add tests** that the expected associations gain evidence and stay safe.
9. **Document population scope + caveats** on the associations you touch.
10. **Demo**: `make foundation-demo` with an intake that should surface the new evidence.

## Worked examples (already in-tree)

| Dataset | Evidence type | Strengthens |
|---|---|---|
| mcPHASES aggregates | `cohort_rate` | symptom ↔ phase `population_enriched_in` edges |
| NHANES harmonized ranges | `reference_range` | FSH/estradiol midlife marker edges |
| Kaggle PCOS model | `model_signal` | `assoc.model_pcos` |
| Menopause-stage model | `model_signal` | `assoc.model_meno` |
| MedQuAD WH subset | `qa_citation` | phrasing context only (never creates edges) |

## If we add EndoLIST tomorrow

1. Place restricted files under `data/endolist/` (gitignored).
2. Add registry entry: longitudinal endometriosis symptom tracking; not redistributable.
3. Map pain/bleeding columns → `sym.pelvic_pain`, `sym.cramps`, cycle timing entities.
4. Strengthen `assoc.pelvic_endo` / `assoc.pelvic_menstrual` with aggregate rates only.
5. Keep strength_prior honest (may stay `low` until evidence is strong).
6. Never emit per-participant rows into `foundation_v0.1.json`.

## If we add real SWAN tomorrow

1. Place `data/swan/swan_harmonized.csv`.
2. Retrain menopause model (`make train-models`).
3. Rebuild foundation — `assoc.model_meno` evidence will flip off synthetic disclaimer.
4. Optionally attach trajectory aggregates to midlife associations.

## Rules of the road

- **Seed = medical prior.** Evidence = measurement.
- **LLM = mouth/ears only.**
- **Restricted data stays local; only aggregates/models ship.**
- Prefer fewer high-quality evidenced edges over a noisy encyclopedia.
