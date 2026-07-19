# CycleBench Medical Foundation

The foundation is the intellectual center of **CycleBench** (used by **Aestra**): a
**structured medical knowledge substrate** that female-specialized datasets plug into as
evidence.

## What it is

```
Entities ── Associations (guideline/textbook seed) ── Evidence (datasets/models)
                              │
                              ▼
                    assemble_read(intake)
                              │
              foundation fact + dataset evidence + your pattern
```

It is **not** an LLM that knows medicine, and **not** a single foundation model.
The LLM never creates edges. Datasets never invent clinical associations alone —
they strengthen existing ones with rates, ranges, and model signals.

## Objects

| Object | Role |
|---|---|
| **Entity** | Symptom, hormone, phase, intervention, confounder, discussion cluster, … |
| **Association** | Typed edge with source, caveats, match tags, talking point, ask-doctor question |
| **Evidence** | Dataset attachment: cohort_rate, reference_range, model_signal, guideline_seed, qa_citation |
| **FoundationBundle** | Versioned graph JSON at `data/foundation/foundation_v0.1.json` |

Discussion **clusters** (PCOS, menopause transition, endometriosis-ish, …) are
explicitly `not_a_diagnosis=true`.

## Rebuild

```bash
make reference      # optional: refresh mcPHASES aggregates first
make foundation     # seed + attach evidence → data/foundation/foundation_v0.1.json
make foundation-demo
```

## How chat uses it

`POST /analyse/feeling-off` calls `assemble_read(intake, personal_findings)`.

Each card is:
1. **Foundation fact** (seed talking point)
2. **Dataset evidence** (mcPHASES rates, NHANES ranges, …)
3. **Your pattern** (deterministic engine finding, when relevant)
4. **Ask your doctor** question
5. Source + dataset tags

## Safety

Every foundation fact and evidence summary passes `cyclebench.safety.assert_safe`.
Language is association / counseling only.

## Growth

See `docs/ADDING_A_DATASET.md` — new datasets become evidence adapters, not rewrites
of the chat or the graph core.
