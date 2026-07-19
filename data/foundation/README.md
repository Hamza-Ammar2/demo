# Medical foundation bundle

Versioned graph: `foundation_v0.1.json`

## License

- **Bundle packaging / schema / seed structure:** CC-BY-4.0 (same as project docs).
- **Code that builds it** (`cyclebench/foundation/`): MIT.
- **Evidence rows:** inherit the source license. Restricted sources contribute **aggregates /
  model signals only** (never raw mcPHASES rows). See each evidence object's `license_note`.

## Rebuild

```bash
make foundation
make foundation-demo
```

## Extend

See `docs/FOUNDATION.md` and `docs/ADDING_A_DATASET.md`.
