---
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
