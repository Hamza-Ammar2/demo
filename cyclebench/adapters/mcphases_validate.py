"""Real mcPHASES validation (aggregate output only).

Grounds CycleBench in real longitudinal data. We answer two questions on the
mcPHASES self-report table WITHOUT redistributing any restricted data:

  Q1. Do headache/fatigue episodes cluster by menstrual-cycle phase across real
      participants? (validates the premise of the pattern detector)
  Q2. Does CycleBench's independent cycle-alignment engine (which derives phase
      from bleeding onsets alone) agree with mcPHASES' hormone-based phase labels?
      (validates the engine on real onsets)

Plus a sleep-confounder overlap rate — the same confounder the engine flags for Sarah.

LICENSE: mcPHASES is restricted (PhysioNet DUA). This module reads it locally and
writes ONLY aggregate statistics to /results. No per-participant rows are emitted.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MCPHASES = ROOT / "data" / "mcphases" / "hormones_and_selfreport.csv"
OUT = ROOT / "results" / "mcphases_validation.json"

# Ordinal symptom scale -> numeric.
ORDINAL = {
    "Not at all": 0, "Very Low/Little": 1, "Low": 2,
    "Moderate": 3, "High": 4, "Very High": 5,
}
EPISODE_THRESHOLD = 4  # "High" or "Very High" counts as an episode.

# mcPHASES (Mira) phase label -> CycleBench engine phase.
MIRA_TO_ENGINE = {
    "Menstrual": "menstrual",
    "Follicular": "follicular",
    "Fertility": "periovulatory",
    "Luteal": "luteal",
}
ENGINE_PHASE_ORDER = ["menstrual", "follicular", "periovulatory", "luteal"]

_BASE = date(2000, 1, 1)


def _to_num(v):
    if v in ORDINAL:
        return ORDINAL[v]
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _is_bleeding(flow) -> bool:
    import math
    if flow is None:
        return False
    if isinstance(flow, float) and math.isnan(flow):
        return False
    return str(flow) not in ("Not at all", "nan")


def run_validation() -> dict:
    import pandas as pd
    from scipy.stats import chisquare

    from cyclebench.engine.cycle import assign_cycle_position
    from cyclebench.schema import AnalysisMode

    if not MCPHASES.exists():
        raise FileNotFoundError(
            f"mcPHASES not found at {MCPHASES}. It is restricted; obtain it from PhysioNet "
            "(see docs/DATASETS.md) and place it under data/mcphases/."
        )

    df = pd.read_csv(MCPHASES)
    df = df[df["phase"].notna()].copy()

    # ---- Q1: symptom episodes by phase (aggregate over all participant-days) ----
    results_by_symptom = {}
    for symptom in ("headaches", "fatigue"):
        num = df[symptom].map(_to_num)
        valid = num.notna()
        sub = df[valid].copy()
        sub["_epi"] = (num[valid] >= EPISODE_THRESHOLD).astype(int)

        days_per_phase = sub.groupby("phase").size()
        epi_per_phase = sub.groupby("phase")["_epi"].sum()
        rate_per_phase = (epi_per_phase / days_per_phase).round(4)

        total_epi = int(sub["_epi"].sum())
        # Expected episodes per phase under the null (uniform rate), weighted by days.
        expected = (days_per_phase / days_per_phase.sum()) * total_epi
        phases = [p for p in ["Menstrual", "Follicular", "Fertility", "Luteal"]
                  if p in epi_per_phase.index]
        observed = [int(epi_per_phase[p]) for p in phases]
        exp = [float(expected[p]) for p in phases]
        chi2 = p = None
        if total_epi >= 20 and all(e > 0 for e in exp):
            chi2, p = chisquare(f_obs=observed, f_exp=exp)

        dominant = rate_per_phase.idxmax() if len(rate_per_phase) else None
        results_by_symptom[symptom] = {
            "n_days_with_report": int(valid.sum()),
            "n_episodes": total_epi,
            "episode_rate_by_phase": rate_per_phase.to_dict(),
            "dominant_phase": dominant,
            "chi2": round(float(chi2), 3) if chi2 is not None else None,
            "p_value": round(float(p), 5) if p is not None else None,
        }

    # ---- Sleep confounder overlap: episode-days that also report high sleep issues ----
    hnum = df["headaches"].map(_to_num)
    snum = df["sleepissue"].map(_to_num)
    epi_mask = hnum >= EPISODE_THRESHOLD
    n_epi_days = int(epi_mask.sum())
    n_epi_with_sleep = int(((snum >= EPISODE_THRESHOLD) & epi_mask).sum())
    sleep_overlap = round(n_epi_with_sleep / n_epi_days, 4) if n_epi_days else None

    # ---- Q2: engine cycle-alignment vs Mira phase labels ----
    agree = total = 0
    per_participant_enrichment = []
    for pid, g in df.groupby("id"):
        g = g.sort_values("day_in_study")
        # Derive bleeding onsets: first bleeding day of each run.
        onsets = []
        prev_bleed = False
        for _, row in g.iterrows():
            bleed = _is_bleeding(row.get("flow_volume"))
            if bleed and not prev_bleed:
                onsets.append(_BASE + timedelta(days=int(row["day_in_study"])))
            prev_bleed = bleed
        if len(onsets) < 2:
            continue

        # Compare engine phase to Mira label for each labeled day.
        for _, row in g.iterrows():
            mira = MIRA_TO_ENGINE.get(row["phase"])
            if mira is None:
                continue
            d = _BASE + timedelta(days=int(row["day_in_study"]))
            pos = assign_cycle_position(d, onsets, AnalysisMode.retrospective)
            if pos.phase is None:
                continue
            total += 1
            if pos.phase == mira:
                agree += 1

        # Per-participant: are this person's headache episodes enriched in one phase?
        gnum = g["headaches"].map(_to_num)
        emask = gnum >= EPISODE_THRESHOLD
        if emask.sum() >= 3:
            epi_phase = g[emask.values]["phase"].value_counts()
            share = g["phase"].value_counts(normalize=True)
            dom = epi_phase.idxmax()
            enrich = (epi_phase[dom] / emask.sum()) / share.get(dom, 1e-9)
            per_participant_enrichment.append({"dominant_phase": dom, "enrichment": float(enrich)})

    engine_vs_mira_agreement = round(agree / total, 4) if total else None
    n_enriched = sum(1 for e in per_participant_enrichment if e["enrichment"] > 1.2)

    out = {
        "dataset": "mcPHASES v1.0.0 (PhysioNet, restricted — aggregate stats only)",
        "n_participants": int(df["id"].nunique()),
        "n_participant_days": int(len(df)),
        "episode_threshold": "High or Very High (>=4 on 0-5 ordinal)",
        "symptom_phase_clustering": results_by_symptom,
        "sleep_confounder": {
            "n_headache_episode_days": n_epi_days,
            "n_also_high_sleep_issue": n_epi_with_sleep,
            "overlap_fraction": sleep_overlap,
        },
        "engine_cycle_alignment_validation": {
            "description": "CycleBench engine phase (from bleeding onsets) vs mcPHASES Mira label",
            "n_days_compared": total,
            "agreement": engine_vs_mira_agreement,
            "n_participants_with_enough_cycles": len(per_participant_enrichment),
        },
        "per_participant_symptom_enrichment": {
            "n_participants_analyzed": len(per_participant_enrichment),
            "n_enriched_gt_1_2x": n_enriched,
            "note": "counts only; no per-participant identifiers or rows are emitted",
        },
    }
    return out


def main() -> int:
    out = run_validation()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nAggregate results written to {OUT.relative_to(ROOT)} (no raw mcPHASES data emitted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
