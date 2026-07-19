"""Load / save foundation bundles."""

from __future__ import annotations

import json
from pathlib import Path

from cyclebench.foundation.schema import FoundationBundle

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION_DIR = ROOT / "data" / "foundation"
DEFAULT_PATH = FOUNDATION_DIR / "foundation_v0.1.json"


def save_bundle(bundle: FoundationBundle, path: Path | None = None) -> Path:
    path = path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2))
    return path


def load_bundle(path: Path | None = None, rebuild_if_missing: bool = True) -> FoundationBundle:
    path = path or DEFAULT_PATH
    if not path.exists():
        if not rebuild_if_missing:
            raise FileNotFoundError(path)
        from cyclebench.foundation.build import build_foundation
        return build_foundation(path=path)
    return FoundationBundle.model_validate_json(path.read_text())
