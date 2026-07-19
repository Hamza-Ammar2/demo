"""CLI entry for Hamza's pFL experiments (optional Track 02 research).

Usage:
  PFL_ROUNDS=3 .venv/bin/python -m federated_learning.run multi_symptom
  make pfl-smoke
"""

from __future__ import annotations

import argparse
import sys


EXPERIMENTS = {
    "multi_site": "federated_learning.federated_personalized_model",
    "patient": "federated_learning.patient_centric_federated_model",
    "temporal": "federated_learning.temporal_patient_federated_model",
    "multi_symptom": "federated_learning.multi_symptom_federated_model",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pFL research experiments")
    parser.add_argument(
        "experiment",
        choices=sorted(EXPERIMENTS),
        help="Which walkthrough experiment to run",
    )
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is required. Install with: pip install 'torch>=2.2' matplotlib", file=sys.stderr)
        return 2

    mod_name = EXPERIMENTS[args.experiment]
    mod = __import__(mod_name, fromlist=["main"])
    mod.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
