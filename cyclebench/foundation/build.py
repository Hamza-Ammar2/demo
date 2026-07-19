"""Build the versioned foundation bundle (seed + evidence)."""

from __future__ import annotations

from pathlib import Path

from cyclebench.foundation.evidence import attach_all_evidence
from cyclebench.foundation.io import DEFAULT_PATH, save_bundle
from cyclebench.foundation.seed import build_seed_bundle
from cyclebench.foundation.schema import FoundationBundle


def build_foundation(path: Path | None = None) -> FoundationBundle:
    bundle = build_seed_bundle()
    bundle = attach_all_evidence(bundle)
    # re-validate integrity after evidence attach
    bundle = FoundationBundle.model_validate(bundle.model_dump())
    save_bundle(bundle, path or DEFAULT_PATH)
    return bundle


def main() -> int:
    b = build_foundation()
    print(
        f"Foundation {b.version}: {len(b.entities)} entities, "
        f"{len(b.associations)} associations, {len(b.evidence)} evidence records"
    )
    print(f"Wrote {DEFAULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
