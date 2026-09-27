# Android Malware Detection — CNS Project

> **Static + Dynamic feature fusion for multi-class Android malware classification, with cross-dataset generalization testing.**

---

## Overview

This project classifies Android applications into malware categories using two kinds of evidence collected from the same apps:

- **Static evidence** — what an app *declares* it can do (permissions, intents, API calls — extracted from the manifest).
- **Dynamic evidence** — what an app *actually does* when run (system calls, binder/IPC activity — captured at runtime).

The model is trained on one dataset and validated on a second, independently collected dataset to measure real generalization — not just in-distribution performance.

---

## Datasets

| Dataset | Role | Apps | Classes | Static features | Dynamic features |
|---|---|---|---|---|---|
| **MalDroid2020** | Train & develop | 11,598 | 5 | 50,621 | 139 / 470 |
| **AndMal2020** | External validation only | 357,805 | 14 + benign | 9,503 | 141 |

> ⚠️ Raw dataset files are **not committed to this repository** (sizes reach ~600 MB per file). See [Dataset Setup](#dataset-setup) below.

### MalDroid2020 — 5 Classes
`Benign · Adware · Banking · SMS Malware · Riskware`

Class counts verified against the official Mahdavifar et al. (2020, IEEE DASC) paper:
`1,253 / 2,100 / 3,904 / 2,546 / 1,795`

### AndMal2020 — External Validation
Only the 5 categories that map to MalDroid's taxonomy are used in the main comparison:

| AndMal category | MalDroid equivalent | Rows used |
|---|---|---|
| Adware | Adware | 47,210 |
| Trojan-Banker | Banking | 887 |
| Trojan-SMS | SMS Malware | 3,125 |
| Riskware | Riskware | 97,349 |
| Benign | Benign | 162,181 |

The remaining 9 AndMal categories (Ransomware, Backdoor, Scareware, etc.) have no MalDroid equivalent and are **excluded** from the main comparison rather than force-fitted.

---

## Proposed Architecture

```
MalDroid2020
    └─► Integrity + Label + Identity Checks
    └─► Leakage-safe Train / Val / Test Split  (grouped by hash where available)
    └─► Train-only Preprocessing               (scaling, feature selection — NEVER fit on val/test/AndMal)
         ├─► Static View   (50,621 → selected subset)
         ├─► Dynamic View  (470 → selected subset)
         └─► Fused View    (Static + Dynamic concatenated)
              └─► Model-specific feature adapters
                   ├─► QDA
                   ├─► NuSVC
                   ├─► LightGBM
                   ├─► Balanced Random Forest
                   └─► Autoencoder
                        └─► Internal Evaluation  (MalDroid held-out)
                        └─► External Validation  (AndMal2020 — used ONCE, after everything is frozen)
```

See [`Docs/05_Proposed_Model_Architecture.md`](Docs/05_Proposed_Model_Architecture.md) for the full diagram.

---

## Experiments

| Experiment | Input features |
|---|---|
| A — Static only | 50,621 raw → selected subset |
| B — Dynamic (syscall only) | 139 raw → selected subset |
| C — Dynamic (syscall + binder) | 470 raw → selected subset |
| D — Static + Dynamic fusion | Selected A + selected C combined |

All 5 models run across all 4 experiments (where mathematically feasible).

---

## Evaluation Metrics

Because both datasets are imbalanced (MalDroid 3.1×, AndMal up to 145×), **accuracy alone is not the headline metric.**

- **Primary:** Macro-F1
- Weighted-F1, Balanced Accuracy
- Per-class Precision / Recall / F1
- Confusion Matrix
- ROC-AUC / PR-AUC (one-vs-rest)
- SHAP values for explainability

Internal (MalDroid held-out) and external (AndMal2020) results are **always reported separately — never averaged.**

---

## Leakage Prevention

The single most critical rule: **everything that touches the data is fit on training data only.**

- Scalers, feature selectors, PCA — fit on train, applied to val/test/AndMal.
- Identity-grouped splitting — same real-world app never appears in two splits (where a hash is available).
- AndMal2020 is touched exactly **once**, after all models and hyperparameters are fully frozen.
- All random operations use fixed, recorded seeds.
- Every fitted artifact is saved with a version tag.

Full protocol: [`Docs/07_Leakage_Reproducibility_Protocol.md`](Docs/07_Leakage_Reproducibility_Protocol.md)

---

## Documentation Index

| # | Document | Contents |
|---|---|---|
| 01 | [Dataset Selection & Problem Statement](Docs/01_Dataset_Selection_and_Problem_Statement.md) | Why we switched from CIC-AndMal2017, dataset rationale |
| 02 | [Deep Dataset Forensic Findings](Docs/02_Deep_Dataset_Forensic_Findings.md) | Verified facts about both datasets (row counts, duplicates, header issues) |
| 03 | [Dataset Comparison & Label Mapping](Docs/03_Dataset_Comparison_and_Label_Mapping.md) | Side-by-side table, AndMal→MalDroid category mapping |
| 04 | [Proposed Data Preprocessing Pipeline](Docs/04_Proposed_Data_Preprocessing_Pipeline.md) | Steps 1–12: static, dynamic, and fusion branches with per-branch diagrams |
| **05** | **[Proposed Model Architecture](Docs/05_Proposed_Model_Architecture.md)** ⬅ _start here for the full picture_ | All 18 pipeline steps, all 5 models, complete end-to-end diagram — the single system overview |
| 06 | [Experimental & Evaluation Plan](Docs/06_Experimental_and_Evaluation_Plan.md) | 4 feature experiments × 5 models, metrics |
| 07 | [Leakage & Reproducibility Protocol](Docs/07_Leakage_Reproducibility_Protocol.md) | Anti-leakage rules, splitting strategy, seed policy |
| 08 | [Limitations & Open Questions](Docs/08_Limitations_and_Open_Questions.md) | Confirmed limitations, open questions, known gaps |


---

## Dataset Setup

Raw dataset files are **not tracked by git** (they are too large and must be obtained from the dataset authors).

Full setup instructions, required paths, and class mappings: **[Docs/DATASET_SETUP.md](Docs/DATASET_SETUP.md)**

**Quick summary:**

```
Dataset/
├── MalDroid2020/Static/feature_vectors_static.csv       ← 568 MB
├── MalDroid2020/Dynamic/feature_vectors_syscalls_*.csv  ← 14 MB + 11 MB
└── AndMal2020/Static/ + AndMal2020/Dynamic/             ← ~6.5 GB
```

> ⚠️ `Dataset/CICInvestAndMal2019/` (if present locally, ~919 MB) is **not used** by this pipeline.

---

## Getting Started — Clone & Data

```bash
# 1. Clone
git clone <repository-url>
cd App-Malware-CNS-Project

# 2. Install Git LFS (one-time per machine)
git lfs install

# 3. Pull large shared artifacts (static: 79 MB, fused: 87 MB)
git lfs pull

# 4. Install Python dependencies
pip install -r requirements.txt
```

After cloning, all Group 3 shared artifacts are available under `runs/real/` — **other model groups do not need to run the pipeline**. See [Docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md](Docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md).

For a full explanation of what is stored in Git, Git LFS, and external storage, see **[Docs/GITHUB_DATA_AND_ARTIFACTS.md](Docs/GITHUB_DATA_AND_ARTIFACTS.md)**.

---

## Known Limitations

1. MalDroid2020 has **no sample hash/ID** — perfect identity-based splitting is not possible for it.
2. AndMal2020's static and dynamic files describe **different apps** — no fusion is possible for AndMal.
3. Only **5 of AndMal's 14 categories** map to MalDroid's taxonomy; the rest are out of scope.
4. We cannot check for **MalDroid ↔ AndMal app overlap** (no hash in MalDroid).
5. MalDroid's static file **mixes numeric and text columns** — naive numeric-only loading will crash.

Full details: [`Docs/08_Limitations_and_Open_Questions.md`](Docs/08_Limitations_and_Open_Questions.md)

---

## Status

> ✅ **Implementation complete.** Group 3 shared pipeline and LightGBM model have been fully executed on the real dataset (`runs/real/`). Final results are verified and documented in `docs/final/`.

---

## Group 3 — Data Pipeline & LightGBM

Group 3 is responsible for the **complete shared data pipeline** and the **LightGBM classifier**.

**Group 3 owns:**
- Data auditing and integrity validation
- Canonical manifest creation
- Leakage-safe train/validation/test splitting
- Feature typing (50,620 static columns classified)
- Static preprocessing (StandardScaler, train-only)
- Dynamic preprocessing (syscall-139 and syscall+Binder-470)
- Static/dynamic feature fusion
- Multi-method feature selection (mutual_info, chi2, LightGBM importance)
- Imbalance handling (class weights + SMOTE artifacts for downstream groups)
- Model-ready shared artifacts for all model groups
- LightGBM implementation, training, evaluation, and model locking

Other model groups (QDA, NuSVC, Balanced RF, Autoencoder) consume Group 3's shared artifacts — they do **not** rebuild the pipeline independently.

### Final LightGBM Result

| Metric | Value |
|---|---|
| **Test Accuracy** | **0.9819121447028424** |
| **Test Macro-F1** | **0.9794806809300622** |
| Test Balanced Accuracy | 0.9809595702907510 |
| Modality | fused (static + dynamic-470) |
| Features | 270 selected |
| Model artifact | `runs/real/artifacts/stage14_frozen_model.joblib` |

---

## Group 3 Final Documentation

### Core Pipeline

| Document | Description |
|---|---|
| [01 — Final Data Pipeline](Docs/final/01_FINAL_DATA_PIPELINE.md) | End-to-end pipeline: all 15 stages, integrity, splitting, preprocessing, fusion, selection |
| [06 — Methodology](Docs/final/06_METHODOLOGY.md) | Academic-style methodology covering all design decisions |
| [07 — Presentation Architecture](Docs/final/07_PRESENTATION_ARCHITECTURE.md) | Slide-ready content with Mermaid diagrams for all pipeline components |

### Group 3 Results

| Document | Description |
|---|---|
| [02 — LightGBM Results](Docs/final/02_GROUP3_LIGHTGBM_RESULTS.md) | Full LightGBM config, per-class metrics, confusion matrix, modality comparison |
| [05 — Final Results Summary](Docs/final/05_FINAL_RESULTS_SUMMARY.md) | 5-minute executive summary of all results with embedded figures |
| [04 — Dataset Limitations](Docs/final/04_DATASET_LIMITATIONS.md) | Data quality disclosures, cross-class duplicates, scale limitations |
| [03 — External Validation / AndMal](Docs/final/03_EXTERNAL_VALIDATION_ANDMAL.md) | AndMal2020 validation attempt, feature-space incompatibility, exact finding |

### Generated Figures

13 publication-quality figures from the real run are in [`Docs/final/assets/figures/`](Docs/final/assets/figures/):

| Figure | Description |
|---|---|
| `fig01_confusion_matrix_fused_test.png` | Final confusion matrix (fused model, TEST split) |
| `fig02_per_class_metrics_fused_test.png` | Per-class Precision / Recall / F1 |
| `fig03_modality_comparison.png` | Accuracy + Macro-F1 across all 4 modalities |
| `fig04_class_distribution_splits.png` | Train / Val / Test class distributions |
| `fig05_feature_type_distribution.png` | Static feature type breakdown (50,620 cols) |
| `fig06_feature_selection_reduction.png` | Feature count reduction to 270 |
| `fig07_stage_execution_times.png` | Execution time per pipeline stage |
| `fig08_class_weights.png` | Class weights used in LightGBM |
| `fig09_modality_metrics_comparison.png` | 3-metric comparison across modalities |
| `fig10_static_dynamic_feature_split.png` | Static vs Dynamic contribution in 270 features |
| `fig11_maldroid_class_distribution.png` | MalDroid2020 full class distribution |
| `fig12_pipeline_durations_horizontal.png` | All 15 stage durations (horizontal) |
| `fig13_top20_selected_features.png` | Top 20 selected features by origin |

---

## For Other Model Groups

Groups 1, 2, 4, and 5 should use the **shared Group 3 preprocessing outputs** rather than independently rebuilding the data pipeline. Using the shared artifacts guarantees all models are evaluated on the same canonical split with the same preprocessing.

**Integration guide:** [Docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md](Docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md)

---

## References

- Mahdavifar et al., *MalDroid2020*, IEEE DASC 2020
- Mahindru & Sangal, *AndMal2020*, 2021
- Ke et al., *LightGBM*, NeurIPS 2017
- Chen, Liaw & Breiman, *Balanced Random Forest*, UC Berkeley Tech Report 666, 2004
- Schölkopf et al., *NuSVC*, Neural Computation 2000
- Lundberg & Lee, *SHAP*, NeurIPS 2017
