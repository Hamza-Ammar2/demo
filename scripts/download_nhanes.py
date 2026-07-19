"""Download open NHANES components used by CycleBench (US public domain, no auth).

CDC serves SAS-transport (.xpt) files at:
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start_year}/DataFiles/{FILE}.xpt

We focus on components relevant to women's hormonal health. Raw .xpt land in
data/raw/ and are converted to data/processed/*.csv (both gitignored; only the
harmonized export under data/nhanes_harmonized/ is published).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"

# (file, cycle_start_year, description)
COMPONENTS = [
    ("P_DEMO", 2017, "Demographics (2017-Mar2020 pre-pandemic)"),
    ("P_TST", 2017, "Sex steroid hormones: testosterone, estradiol, SHBG"),
    ("P_BMX", 2017, "Body measures (BMI, waist)"),
    ("P_BIOPRO", 2017, "Standard biochemistry profile"),
    ("P_RHQ", 2017, "Reproductive health: menarche, menstrual regularity, menopause, pregnancies, HRT"),
    ("P_GHB", 2017, "Glycohemoglobin (HbA1c)"),
    ("P_GLU", 2017, "Fasting plasma glucose"),
    ("P_INS", 2017, "Insulin"),
    ("P_SLQ", 2017, "Sleep disorders / sleep hours"),
    ("P_DPQ", 2017, "Depression screener (PHQ-9)"),
    ("THYROD_G", 2011, "Thyroid profile: TSH, free/total T3+T4, TPO & Tg antibodies (2011-2012)"),
]


def download() -> None:
    import urllib.request

    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    for name, year, desc in COMPONENTS:
        url = BASE.format(year=year, name=name)
        dest = RAW / f"{name}.xpt"
        try:
            print(f"↓ {name}: {desc}")
            urllib.request.urlretrieve(url, dest)
            head = dest.read_bytes()[:64]
            if b"<html" in head.lower():
                print(f"  FAILED (got HTML, not data): {url}")
                dest.unlink(missing_ok=True)
                continue
            df = pd.read_sas(dest, format="xport")
            df.to_csv(PROC / f"{name}.csv", index=False)
            print(f"  ok — {df.shape[0]} rows x {df.shape[1]} cols")
        except Exception as e:  # noqa: BLE001
            print(f"  error: {e}")


if __name__ == "__main__":
    download()
