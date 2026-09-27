# Dataset Setup
## Android Malware Classification — Group 3

> **Raw datasets are NOT committed to this repository.** This document explains what datasets are required, where to obtain them, and how to structure them locally.

---

## Required Datasets

The pipeline uses **two** datasets:

| Dataset | Role | Samples | Size (approx) |
|---|---|---|---|
| **MalDroid2020** | Primary — all model development | 11,598 apps | ~595 MB |
| **AndMal2020** | External generalization analysis only | 357,805+ records | ~6.5 GB |

A third dataset (`CICInvestAndMal2019`) is present in some local copies but is **not used** by this pipeline.

---

## MalDroid2020

**Citation:** Mahdavifar et al., *Deep and Transfer Learning for Android Malware Detection*, IEEE DASC 2020.

**Official source:** IEEE DataPort / paper supplementary.  
**Contact the dataset authors** or search IEEE DataPort for "MalDroid2020".

### Required files

Place these files at the **exact paths below** (relative to the repository root):

```
Dataset/
└── MalDroid2020/
    ├── Static/
    │   └── feature_vectors_static.csv           ← 568 MB, raw static features (50,620 cols)
    └── Dynamic/
        ├── feature_vectors_syscalls_frequency_5_Cat.csv        ← 14 MB, Dynamic-139
        └── feature_vectors_syscallsbinders_frequency_5_Cat.csv ← 11 MB, Dynamic-470
```

### Known issues with the raw static file

The raw `feature_vectors_static.csv` has header/encoding issues that the pipeline repairs automatically in Stage 01/02. The pipeline generates `feature_vectors_static_repaired.parquet` at runtime. Do not manually edit the raw CSV.

### Verified row counts

| File | Rows | Columns | Classes |
|---|---|---|---|
| Static CSV | 11,598 | 50,620 | 5 |
| Dynamic-139 CSV | 11,598 | 141 (incl. label+split cols) | 5 |
| Dynamic-470 CSV | 11,598 | 472 (incl. label+split cols) | 5 |

### Class mapping

| Integer | Class |
|---|---|
| 1 | Adware |
| 2 | Banking |
| 3 | SMS Malware |
| 4 | Riskware |
| 5 | Benign |

---

## AndMal2020

**Citation:** Mahindru & Sangal, *MLDroid — Framework for Android Malware Detection*, Neural Computing and Applications, 2021.

**Official source:** Kaggle / paper supplementary.  
**Contact the dataset authors** or search for "AndMal2020 malware dataset".

### Required files

```
Dataset/
└── AndMal2020/
    ├── Static/
    │   ├── Benign/
    │   │   ├── Ben0.csv    (583 MB)
    │   │   ├── Ben1.csv    (870 MB)
    │   │   ├── Ben2.csv    (775 MB)
    │   │   ├── Ben3.csv    (142 MB)
    │   │   └── Ben4.csv    (577 MB)
    │   └── Malicious/
    │       ├── Adware.csv     (857 MB)
    │       ├── Banker.csv     (16 MB)
    │       ├── Backdoor.csv   (28 MB)
    │       ├── Dropper.csv    (42 MB)
    │       ├── FileInfector.csv (12 MB)
    │       ├── NoCategory.csv (42 MB)
    │       ├── PUA.csv        (37 MB)
    │       ├── Ransomware.csv (112 MB)
    │       ├── Riskware.csv   (1768 MB)
    │       ├── Scareware.csv  (28 MB)
    │       ├── SMS.csv        (57 MB)
    │       ├── Spy.csv        (64 MB)
    │       ├── Trojan.csv     (246 MB)
    │       └── Zeroday.csv    (242 MB)
    └── Dynamic/
        └── BeforeAndAfterReboot/
            ├── Adware_before_reboot_Cat.csv
            ├── Adware_after_reboot_Cat.csv
            └── ... (28 CSV files total, ~25 MB)
```

### Usage in this project

- AndMal2020 is used **only for external generalization analysis** (Stage 15) — never for training, validation, or preprocessing fitting
- Only 5 of 14 AndMal categories map to MalDroid's 5-class taxonomy; unmappable categories are excluded
- The AndMal dynamic representation is **incompatible** with the locked fused LightGBM model's feature space (different instrumentation framework — see `docs/final/03_EXTERNAL_VALIDATION_ANDMAL.md`)

---

## CICInvestAndMal2019 — NOT USED

If you have `Dataset/CICInvestAndMal2019/` locally (~919 MB, 2,143 files), it is **not referenced by any pipeline stage** in this project. It can be safely excluded.

---

## Post-Download Setup

Once dataset files are in place, verify setup by running:

```powershell
python one_run.py --config config/smoke.yaml   # fast smoke test (~30s)
```

For the full run:

```powershell
python one_run.py --config config/default.yaml
```

The pipeline will skip already-completed stages (config-hash based resumability via `runs/real/status.json`).

---

## What the Pipeline Produces From Raw Data

```
Raw CSV files
    ↓ Stage 01: audit + repair
    ↓ Stage 02: canonical manifests
    ↓ Stage 03: feature typing (50,620 → 31,652 usable)
    ↓ Stage 04: leakage-safe split (seed 42, 80/10/10)
    ↓ Stage 05: static preprocessing (StandardScaler, train-only)
    ↓ Stage 06: dynamic preprocessing (StandardScaler, train-only)
    ↓ Stage 07: fusion (static + dynamic-470)
    ↓ Stage 08: feature selection (K ∈ {50,100,270})
    ↓ Stage 09: imbalance handling (class weights + SMOTE)
    ↓ Stage 11–14: LightGBM training + evaluation + model lock
    ↓ Stage 15–16: external validation + final report
        → runs/real/  (all shareable artifacts committed to Git/LFS)
```

All shareable artifacts from `runs/real/` are already committed to this repository (via Git / Git LFS). Other model groups do **not** need to run the pipeline — they can use the committed artifacts directly. See [`docs/08_MODEL_GROUP_INTEGRATION_GUIDE.md`](08_MODEL_GROUP_INTEGRATION_GUIDE.md).
