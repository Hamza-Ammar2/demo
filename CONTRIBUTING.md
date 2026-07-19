# Contributing to CycleBench / Aestra

Thank you for extending the open stack. Prefer small, reviewable PRs that leave the next
person a clear way to reproduce your change.

## Before you open a PR

```bash
make install
make test
```

If you change engine thresholds, schema fields, or safety copy:

```bash
make benchmark
make foundation   # if seed evidence / graph changed
```

## What makes a strong contribution (hackathon + beyond)

Aligned with open AI infrastructure judging:

1. **Reusable artifact** — dataset adapter, schema field, foundation edge, model task, or
   evaluation case — not only a one-off UI tweak.
2. **Transparent methods** — document assumptions, preprocessing, and evaluation choices in
   `docs/` or adjacent README (see `docs/DECISIONS.md` style).
3. **Reproducible** — a Makefile target or script + pinned deps so others can re-run.
4. **Honest licensing** — never commit PhysioNet / DUA-restricted participant rows; ship
   aggregates + access instructions only.
5. **No unsupported medical claims** — keep safety language; do not present the UI as diagnosis.

## Good places to add value

| Contribution | Start here |
|--------------|------------|
| New dataset adapter | `docs/ADDING_A_DATASET.md`, `cyclebench/adapters/` |
| New foundation evidence | `docs/FOUNDATION.md`, `cyclebench/foundation/seed.py` |
| New model task | `docs/MODEL_CARD.md`, `cyclebench/models/` |
| New benchmark case | `docs/BENCHMARK.md`, `cyclebench/benchmark/` |
| Product UX (Aestra) | `web/`, `cyclebench/api/` — keep deterministic engine as source of truth |

## Do not commit

- `.env` or PhysioNet credentials
- Raw mcPHASES / other restricted participant files
- Large binary dumps outside `models/*.joblib` already tracked by design

## License

Code: MIT. Docs / foundation packaging: CC-BY-4.0. Cite via `CITATION.cff`.
