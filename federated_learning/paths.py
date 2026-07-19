"""Shared paths for Hamza's federated_learning experiments (repo-relative)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FL_DIR = Path(__file__).resolve().parent
MCPHASES = ROOT / "data" / "mcphases"
MCP_HORM_PATH = MCPHASES / "hormones_and_selfreport.csv"

# Optional NHANES raw (Experiment 1). Not required for patient-centric exps 2–4.
NHANES_TST = ROOT / "data" / "raw" / "P_TST.csv"
NHANES_DEMO = ROOT / "data" / "raw" / "P_DEMO.XPT"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rounds(default: int = 30) -> int:
    return int(os.environ.get("PFL_ROUNDS", str(default)))


def require_mcphases() -> Path:
    if not MCP_HORM_PATH.exists():
        raise FileNotFoundError(
            f"mcPHASES hormones file missing: {MCP_HORM_PATH}\n"
            "Place PhysioNet extract under data/mcphases/ (gitignored)."
        )
    return MCP_HORM_PATH
