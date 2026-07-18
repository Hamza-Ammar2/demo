"""CycleBench Layer 02 — focused, explainable models.

Two published model tasks:
  1. Multi-source hormonal-state classifier (mcPHASES): wearables + symptoms → cycle phase
  2. Menopause-stage classifier (SWAN public-use / synthetic fallback): hormones + symptoms → stage

Design invariant: models emit probabilities + feature attributions. They never diagnose.
Outputs are wrapped as schema Findings (evidence_class=inferred) for the Case Compiler brief.
"""

from cyclebench.model.predict import load_bundle, predict_hormonal_state, predict_menopause_stage

__all__ = [
    "load_bundle",
    "predict_hormonal_state",
    "predict_menopause_stage",
]
