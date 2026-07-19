"""Dataset registry — what each source can and cannot validly support.

A small, importable source of truth so analyses don't misuse a dataset (e.g. treating
NHANES as longitudinal, or claiming menopause prediction from a young-adult cohort).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    license: str
    redistributable: bool
    longitudinal: bool
    n_subjects: str
    modalities: list
    supports: list          # tasks this dataset can validly support
    does_not_support: list  # tasks it must NOT be used for
    notes: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    "mcphases": DatasetSpec(
        name="mcPHASES v1.0.0 (PhysioNet)",
        license="PhysioNet Restricted Health Data License 1.5.0",
        redistributable=False,
        longitudinal=True,
        n_subjects="42 (20 with a 2nd interval)",
        modalities=["wearable (Fitbit)", "CGM", "urine hormone metabolites (Mira: LH/E3G/PdG)",
                    "self-report symptoms", "cycle phase labels"],
        supports=[
            "cycle-phase / hormonal-STATE classification (labeled)",
            "symptom-vs-phase association analysis",
            "wearable-vs-phase association analysis",
            "within-person longitudinal pattern detection",
            "leakage-audited, participant-split prediction baselines",
        ],
        does_not_support=[
            "quantitative hormone-LEVEL prediction as a clinical output (noisy consumer proxy, n=42)",
            "menopause-onset prediction (cohort is young reproductive-age; no peri/post-menopausal)",
            "generalization beyond young, mostly-healthy menstruators without hormonal contraception",
        ],
        notes="Hormone values are Mira urine metabolites (semi-quantitative), manually entered.",
    ),
    "nhanes": DatasetSpec(
        name="NHANES 2017-March 2020 (CDC)",
        license="US public domain",
        redistributable=True,
        longitudinal=False,
        n_subjects="thousands (cross-sectional)",
        modalities=["serum sex-steroid panel", "demographics", "body measures", "biochemistry"],
        supports=[
            "population reference ranges (age/sex-stratified)",
            "cross-sectional associations",
            "harmonized open dataset publication",
        ],
        does_not_support=[
            "within-person cycle dynamics (one blood draw per person)",
            "row-wise merging with mcPHASES participants",
            "longitudinal / temporal-leakage tasks",
        ],
        notes="Total testosterone is RDC-only for 2017+ (public only in 2013-2016 cycles).",
    ),
    "swan": DatasetSpec(
        name="SWAN — Study of Women's Health Across the Nation (ICPSR public-use)",
        license="ICPSR terms / study-specific; public-use files available",
        redistributable=False,
        longitudinal=True,
        n_subjects="~3300 midlife women",
        modalities=["serum hormones (E2, FSH, SHBG, …)", "vasomotor symptoms",
                    "sleep", "menstrual/menopausal status"],
        supports=[
            "menopausal stage / transition classification",
            "hormone + symptom trajectories across midlife",
            "explainable multi-source menopause-stage models",
        ],
        does_not_support=[
            "young-adult cycle-phase prediction (wrong age band)",
            "ovarian cyst imaging diagnosis",
            "exact calendar date of final menstrual period as a clinical device output",
        ],
        notes="Place harmonized CSV at data/swan/swan_harmonized.csv; see docs/SWAN_ACCESS.md.",
    ),
    "pcos_kaggle": DatasetSpec(
        name="PCOS dataset (Kaggle: shreyasvedpathak/pcos-dataset)",
        license="Kaggle 'copyright-authors' — not redistributed here (adapter + model only)",
        redistributable=False,
        longitudinal=False,
        n_subjects="541 patients (single-region clinics, Kerala, India)",
        modalities=["clinical vitals", "serum hormones (FSH, LH, AMH, PRL, TSH, β-HCG)",
                    "ultrasound follicle counts", "cycle regularity", "symptom flags"],
        supports=[
            "PCOS-risk association from clinical + hormonal + symptom features",
            "explainable screening signal for research/education",
            "leakage-audited, stratified-split prediction baseline",
        ],
        does_not_support=[
            "clinical diagnosis of PCOS",
            "generalization beyond a similar clinical population",
            "menopause or cycle-phase prediction",
        ],
        notes="Cross-sectional, class-imbalanced (33% positive). Model trained via cyclebench.model.tasks (pcos_risk).",
    ),
}


def can_support(dataset: str, task_keyword: str) -> bool:
    spec = REGISTRY[dataset]
    kw = task_keyword.lower()
    if any(kw in s.lower() for s in spec.does_not_support):
        return False
    return any(kw in s.lower() for s in spec.supports)


def as_markdown() -> str:
    lines = ["# CycleBench Dataset Registry\n"]
    for spec in REGISTRY.values():
        lines.append(f"## {spec.name}")
        lines.append(f"- License: {spec.license} (redistributable: {spec.redistributable})")
        lines.append(f"- Longitudinal: {spec.longitudinal} | Subjects: {spec.n_subjects}")
        lines.append(f"- Modalities: {', '.join(spec.modalities)}")
        lines.append("- **Can support:**")
        lines += [f"  - {s}" for s in spec.supports]
        lines.append("- **Must NOT be used for:**")
        lines += [f"  - {s}" for s in spec.does_not_support]
        if spec.notes:
            lines.append(f"- Notes: {spec.notes}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(as_markdown())
