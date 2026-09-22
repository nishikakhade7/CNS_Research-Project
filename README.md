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
| 05 | [Proposed Model Architecture](Docs/05_Proposed_Model_Architecture.md) | Steps 13–14: end-to-end architecture, model-specific adapters |
| 06 | [Experimental & Evaluation Plan](Docs/06_Experimental_and_Evaluation_Plan.md) | 4 feature experiments × 5 models, metrics |
| 07 | [Leakage & Reproducibility Protocol](Docs/07_Leakage_Reproducibility_Protocol.md) | Anti-leakage rules, splitting strategy, seed policy |
| 08 | [Limitations & Open Questions](Docs/08_Limitations_and_Open_Questions.md) | Confirmed limitations, open questions, known gaps |
| 09 | [Professor Approval Summary](Docs/09_Professor_Approval_Summary.md) | One-page summary + questions for supervisor |
| **10** | **[Complete System Architecture](Docs/10_System_Architecture.md)** ⬅ _start here for the full picture_ | All 18 pipeline steps, all 5 models, one end-to-end diagram — synthesises Docs 04 + 05 |

---

## Dataset Setup

Dataset files are not tracked by git. To reproduce the environment:

1. Download **MalDroid2020** from the official source and place files at:
   ```
   Dataset/MalDroid2020/Static/feature_vectors_static.csv
   Dataset/MalDroid2020/Dynamic/
   ```
2. Download **AndMal2020** and place files at:
   ```
   Dataset/AndMal2020/Static/
   Dataset/AndMal2020/Dynamic/
   ```

Refer to [`Docs/02_Deep_Dataset_Forensic_Findings.md`](Docs/02_Deep_Dataset_Forensic_Findings.md) for known issues with the raw files (missing headers, mixed column types, near-duplicates) that must be handled before any code runs.

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

> 🔬 **Research & planning phase.** No model training has been run yet.
> All documents marked `PROPOSED` describe our intended approach, not implemented code.
> Documents marked `CONFIRMED` reflect facts directly verified from the raw dataset files.

---

## References

- Mahdavifar et al., *MalDroid2020*, IEEE DASC 2020
- Mahindru & Sangal, *AndMal2020*, 2021
- Ke et al., *LightGBM*, NeurIPS 2017
- Chen, Liaw & Breiman, *Balanced Random Forest*, UC Berkeley Tech Report 666, 2004
- Schölkopf et al., *NuSVC*, Neural Computation 2000
- Lundberg & Lee, *SHAP*, NeurIPS 2017
