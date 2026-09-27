# GitHub Data & Artifacts Guide
## Android Malware Classification — Group 3

This document explains **what is stored where** in this repository so you can obtain exactly what you need.

---

## Summary

| Category | Storage | How to Get It |
|---|---|---|
| Pipeline source code | Normal Git | `git clone <repo>` |
| Documentation & figures | Normal Git | `git clone <repo>` |
| Small shared artifacts (< 15 MB) | Normal Git | `git clone <repo>` |
| Large shared feature artifacts (79–87 MB) | **Git LFS** | `git lfs pull` after cloning |
| Raw MalDroid2020 & AndMal2020 datasets | **External** | See [`docs/DATASET_SETUP.md`](DATASET_SETUP.md) |
| Runtime logs / intermediate checkpoints | Local only (ignored) | Re-run pipeline if needed |

---

## Files in Normal Git (no LFS)

These are committed as standard Git objects and are automatically available after `git clone`:

### Mandatory for all model groups

| Path | Size | Purpose |
|---|---|---|
| `runs/real/manifests/maldroid_split_manifest.parquet` | 0.13 MB | Canonical split contract (train=9,281 / val=1,156 / test=1,161) |
| `runs/real/artifacts/stage08_feature_selection.json` | 0.17 MB | Feature lists for all modalities, methods, K values |
| `runs/real/artifacts/stage14_lock_manifest.json` | 0.015 MB | Frozen LightGBM feature contract + verified test metrics |

### Dynamic feature artifacts

| Path | Size | Purpose |
|---|---|---|
| `runs/real/artifacts/stage06_dynamic_139_features.parquet` | 2.52 MB | Preprocessed dynamic-139 features (syscall only) |
| `runs/real/artifacts/stage06_dynamic_470_features.parquet` | 4.35 MB | Preprocessed dynamic-470 features (syscall + Binder) |
| `runs/real/artifacts/stage06_dynamic_139_scaler.joblib` | 0.007 MB | Fitted StandardScaler for dynamic-139 |
| `runs/real/artifacts/stage06_dynamic_470_scaler.joblib` | 0.028 MB | Fitted StandardScaler for dynamic-470 |

### Group 3 LightGBM model

| Path | Size | Purpose |
|---|---|---|
| `runs/real/artifacts/stage14_frozen_model.joblib` | 5.06 MB | Frozen final LightGBM (270 features, fused modality) |

### Imbalance / SMOTE

| Path | Size | Purpose |
|---|---|---|
| `runs/real/artifacts/stage09_imbalance_report.json` | 0.25 MB | Class weights per modality |
| `runs/real/artifacts/stage09_smote_dynamic_139.npz` | 3.73 MB | SMOTE-resampled training data (dynamic-139) |
| `runs/real/artifacts/stage09_smote_dynamic_470.npz` | 6.38 MB | SMOTE-resampled training data (dynamic-470) |

### Metrics & reports

| Path | Size | Purpose |
|---|---|---|
| `runs/real/metrics/stage13_ablation_results.json` | 0.007 MB | Modality comparison (all 4 modalities) |
| `runs/real/metrics/stage15_external_validation.json` | 0.001 MB | AndMal external validation result |
| `runs/real/metrics/stage12_internal_evaluation.json` | 0.002 MB | Stage 12 evaluation |
| `runs/real/metrics/stage11_training_results.json` | ~0 MB | Stage 11 train metrics |
| `runs/real/reports/stage01_data_audit.json` | 0.011 MB | Dataset audit report |
| `runs/real/reports/stage04_cross_class_duplicate_conflicts.json` | 0.002 MB | Duplicate-label conflict disclosure |
| `runs/real/reports/FINAL_REPORT.md` | 0.002 MB | Pipeline final report |

### Run metadata

| Path | Size | Purpose |
|---|---|---|
| `runs/real/environment.json` | 0.001 MB | Python packages + GPU info at run time |
| `runs/real/config.yaml.snapshot.json` | 0.002 MB | Exact config used for the real run |
| `runs/real/status.json` | 0.008 MB | Stage completion history (paths sanitized) |

### Manifests

| Path | Size | Purpose |
|---|---|---|
| `runs/real/manifests/maldroid_manifest.parquet` | 0.07 MB | Full MalDroid pre-split manifest |

### Pipeline plots

| Path | Size | Purpose |
|---|---|---|
| `runs/real/plots/confusion_matrix_lightgbm.png` | 0.055 MB | Confusion matrix (pipeline-generated) |
| `runs/real/plots/modality_comparison_lightgbm.png` | 0.031 MB | Modality comparison plot |
| `runs/real/plots/internal_macro_f1_by_model.png` | 0.025 MB | Macro-F1 per modality |
| `runs/real/plots/training_time_by_model.png` | 0.025 MB | Training time plot |

---

## Files in Git LFS

These files are large and stored via Git LFS. After cloning, they appear as **LFS pointer files** (small text placeholders) until you run `git lfs pull`.

| Path | Size | LFS Hash | Purpose |
|---|---|---|---|
| `runs/real/artifacts/stage05_static_features.parquet` | **79.13 MB** | c19ff82bb1 | Preprocessed static features (11,598 × 63,306) |
| `runs/real/artifacts/stage05_static_scaler.joblib` | **6.18 MB** | e722819cb4 | Fitted StandardScaler for static features |
| `runs/real/artifacts/stage07_fused_features.parquet` | **87.34 MB** | 4fca768f95 | Fused features (11,598 × 64,247) |
| `runs/real/manifests/andmal_manifest.parquet` | **14.97 MB** | d95fd6883c | AndMal2020 manifest (357,805 rows) |
| `Docs/Android_Malware_Project_Full_Documentation.docx` | 0.91 MB | — | Full project documentation (binary) |

> **Note:** Git LFS requires a remote LFS server (GitHub.com supports LFS natively with a free tier for public repositories). If you only need the small artifacts and do not need the large static/fused Parquets, you can skip `git lfs pull` — only those 4 files will be missing.

---

## Files Outside the Repository (External / Local Only)

These are **never** committed to Git or LFS:

| Path | Size | Reason |
|---|---|---|
| `Dataset/MalDroid2020/Static/feature_vectors_static.csv` | 568 MB | Raw source — too large, obtain from dataset authors |
| `Dataset/MalDroid2020/Static/feature_vectors_static_repaired.csv` | 568 MB | Derived duplicate of above — regeneratable |
| `Dataset/MalDroid2020/Static/feature_vectors_static_repaired.parquet` | 151 MB | Pipeline input — regeneratable via `one_run.py` |
| `Dataset/MalDroid2020/Dynamic/*.csv` | ~25 MB | Raw source |
| `Dataset/AndMal2020/**/*.csv` | ~6,526 MB | Raw source — obtain from dataset authors |
| `runs/real/logs/` | ~0.05 MB | Runtime logs — machine-specific |
| `runs/real/checkpoints/lightgbm_dynamic470.joblib` | 5.03 MB | Intermediate checkpoint — not final model |
| `runs/real/artifacts/stage03_*.json` | ~8 MB | Regeneratable internal artifacts |
| `runs/smoke/` | ~0.4 MB | Smoke-test run outputs |

---

## Clone Workflow

```bash
# Step 1: Clone the repository
git clone <repository-url>
cd App-Malware-CNS-Project

# Step 2: Initialize LFS (only needed once per machine)
git lfs install

# Step 3: Pull LFS-tracked large artifacts
git lfs pull

# Step 4: Verify LFS files are real data (not pointer text)
python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('runs/real/artifacts/stage07_fused_features.parquet')
print('Fused artifact OK:', pf.metadata.num_rows, 'rows x', pf.metadata.num_columns, 'cols')
"
```

If you only need small artifacts (feature lists, manifests, metrics), skip Step 3 — everything except the 4 LFS files is available immediately after Step 1.

---

## For Model Group Members Who Do Not Need Large Parquets

The following groups may **not** need the large LFS artifacts at all:

| Group | Large Parquets Needed? | Alternative |
|---|---|---|
| Group 1 (QDA) | `stage05` (79 MB) needed | Can re-run Stage 5 from raw data |
| Group 2 (NuSVC) | `stage06` only (small, 2–4 MB) | Already in normal Git |
| Group 3 (LightGBM) | `stage07` (87 MB) for reference | Already have the frozen model |
| Group 4 (Balanced RF) | `stage06`/`stage07` (small–large) | Can use `stage06` only (in Git) |
| Group 5 (Autoencoder) | `stage05` or `stage07` needed | May need LFS pull |

See [`docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md`](08_MODEL_GROUP_INTEGRATION_GUIDE.md) for the complete per-group feature-selection and artifact guide.

---

## Repository Size Summary

| Category | Size | Storage |
|---|---|---|
| Source code + config | ~0.3 MB | Normal Git |
| Documentation + figures | ~2.1 MB | Normal Git |
| Small shared artifacts | ~25 MB | Normal Git |
| Large shared artifacts | ~188 MB | Git LFS |
| Raw datasets | ~8,760 MB | **External** |
| Runtime logs/cache | ~0.5 MB | Local only |
