import os
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Paths
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
EXPORT_DIR = ROOT / "results" / "hf_export"
DATA_DIR = EXPORT_DIR / "data"

# Classes and Features matching McPhases setup
FEATURE_COLS = [
    "headaches_ord", "cramps_ord", "sorebreasts_ord", "fatigue_ord", "sleepissue_ord",
    "moodswing_ord", "stress_ord", "foodcravings_ord", "indigestion_ord", "bloating_ord",
    "appetite_ord", "sleep_minutes", "sleep_awake", "sleep_efficiency", "resting_hr",
    "steps_sum", "stress_score_mean", "wrist_temp_delta", "glucose_mean", "hrv_rmssd", "is_weekend"
]

def main():
    print("=== Step 1: Loading McPhases Feature Table ===")
    from cyclebench.model.features_mcphases import build_mcphases_table
    try:
        df = build_mcphases_table()
    except Exception as e:
        print(f"Error loading McPhases: {str(e)}")
        return
        
    print(f"Loaded {len(df)} total rows.")
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    unique_ids = df["id"].unique()
    print(f"Processing {len(unique_ids)} unique patients...")
    
    # Process each patient and write to data/client_[UUID].csv
    for pid in unique_ids:
        sub_df = df[df["id"] == pid].sort_values(by="day_in_study").copy()
        
        # 1. Generate anonymous, reproducible UUID based on subject ID
        anon_uuid = f"client_{uuid.uuid5(uuid.NAMESPACE_DNS, str(pid)).hex[:12]}"
        
        # 2. Interpolate and pad locally
        sub_df[FEATURE_COLS] = sub_df[FEATURE_COLS].interpolate(method="linear").ffill().bfill().fillna(0.0)
        
        # Keep only required columns
        out_cols = FEATURE_COLS + ["phase"]
        export_df = sub_df[out_cols].copy()
        
        # Save locally
        out_path = DATA_DIR / f"{anon_uuid}.csv"
        export_df.to_csv(out_path, index=False)
        print(f"  - Saved anonymous logs for subject {pid} to {out_path.name}")
        
    # Write Hugging Face Dataset Card (README.md)
    readme_content = """---
language: en
license: openrail
tags:
- medical
- time-series
- wearables
- menstrual-health
pretty_name: CycleBench Wearable Symptom Log Dataset
size_categories:
- 1K<n<10K
---

# CycleBench Wearable Symptom Log Dataset

This dataset contains anonymized, multi-modal symptom and wearable logs collected for menstrual health research. 

It is divided into client-specific files to allow conflict-free updates from federated learning instances.

## Dataset Structure
Each file in the `data/` directory belongs to a unique client (`data/client_[UUID].csv`) containing daily tracking rows:
* **Hormone levels & symptoms**: Headaches, cramps, bloating, etc. (mapped to $0$-$5$ ordinal levels).
* **Wearable logs**: Resting HR, sleep minutes, sleep efficiency, steps, and wrist temperature delta.
* **Target phase**: Inferred cycle phase (`Menstrual`, `Follicular`, `Fertility`, `Luteal`).
"""
    with open(EXPORT_DIR / "README.md", "w") as f:
        f.write(readme_content)
    print("\n=== Step 2: Local Dataset Prepared successfully ===")
    print(f"Local export directory: {EXPORT_DIR}")
    
    # 3. Check if HF Write Credentials are present to upload automatically
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_DATASET_REPO")
    
    if hf_token and hf_repo:
        print(f"\n=== Step 3: Pushing to Hugging Face Dataset Repository '{hf_repo}' ===")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Create repository if it doesn't exist
            print(f"Creating/verifying Hugging Face repository '{hf_repo}'...")
            api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, token=hf_token)
            
            # Upload folder
            print("Uploading dataset files...")
            api.upload_folder(
                folder_path=str(EXPORT_DIR),
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token
            )
            print("Successfully uploaded all files to Hugging Face!")
            print(f"Access it here: https://huggingface.co/datasets/{hf_repo}")
        except Exception as e:
            print(f"Upload failed: {str(e)}")
    else:
        print("\n=== Step 3: Hugging Face upload skipped ===")
        print("To push to Hugging Face, set environment variables:")
        print("  export HF_TOKEN=your_token")
        print("  export HF_DATASET_REPO=username/repo-name")
        print("And run: .venv/bin/python scripts/prepare_hf_dataset.py")

if __name__ == "__main__":
    main()
