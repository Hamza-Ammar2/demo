"""Download curated Kaggle datasets for CycleBench (needs a Kaggle API token).

These are the "feed the AI" datasets that require authentication (unlike NHANES,
which is open and already downloaded by scripts/download_nhanes.py).

Setup (one time):
  1. kaggle.com -> your profile -> Settings -> API -> "Create New Token"
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
     (or export KAGGLE_USERNAME / KAGGLE_KEY in your environment / .env)
  3. pip install kaggle
  4. python scripts/download_kaggle.py

Nothing here is committed: raw Kaggle data lands in data/kaggle/ (gitignored).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kaggle"

# slug -> why we want it
DATASETS = {
    # Labeled clinical/hormonal PCOS cohort: FSH, LH, TSH, AMH, PRL, follicle counts,
    # cycle regularity + symptoms. Enables a real PCOS-risk prediction task.
    "shreyasvedpathak/pcos-dataset": "PCOS clinical+hormonal (541 patients, labeled)",
    # Consumer medical Q&A pairs — grounding/eval text for the LLM chat layer.
    "thedevastator/comprehensive-medical-q-a-dataset": "Medical Q&A pairs (LLM grounding)",
}


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> int:
    if not have_credentials():
        print("No Kaggle credentials found.\n"
              "  -> Put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY.\n"
              "  -> See the header of this file for steps.")
        return 1
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("The 'kaggle' package is not installed. Run: pip install kaggle")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for slug, why in DATASETS.items():
        dest = OUT / slug.split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"↓ {slug}  — {why}")
        rc = subprocess.call([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", slug, "-p", str(dest), "--unzip",
        ])
        print("  ok" if rc == 0 else f"  FAILED (rc={rc}) — check the slug/access on kaggle.com")
    print(f"\nDone. Files under {OUT.relative_to(ROOT)} (gitignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
