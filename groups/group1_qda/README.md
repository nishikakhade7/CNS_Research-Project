# Group 1 — Quadratic Discriminant Analysis (QDA) Android Malware Detection

This module implements Quadratic Discriminant Analysis (QDA) for Android malware classification on the MalDroid-2020 dataset (5 classes: Adware, Banking, Benign, Riskware, SMS Malware). All cross-group comparison results use the **official shared 3-way split** (Section 1). The 80:20 stratified ablation (Section 2) is a secondary experiment kept for reproducibility but **not used for cross-group comparison**.

---

## 1. Official Results — Shared 3-Way Split (Primary / Cross-Group Comparison)

The headline result uses the project-wide `maldroid_split_manifest.parquet` split:
- **Train**: 9,281 samples
- **Val**: 1,156 samples (regularisation tuning only — never seen by final model until reg_param selected)
- **Test**: 1,161 samples (evaluated exactly once on the final frozen model)

Per the Integration Guide (Section 4): cross-group comparison requires this shared split. The numbers below are the canonical Group 1 submission metrics.

### Overall Metrics

| Metric | Value |
|---|---|
| **Test Accuracy** | **92.51%** (0.9251) |
| **Balanced Accuracy** | **92.90%** (0.9290) |
| **Macro Precision** | **91.03%** (0.9103) |
| **Macro Recall** | **92.90%** (0.9290) |
| **Macro F1-Score** | **91.82%** (0.9182) |
| **Weighted F1-Score** | **92.58%** (0.9258) |
| **Train Accuracy** | 92.49% |
| **Train Macro F1** | 91.90% |
| **Training Runtime** | 0.30 s |
| **Selected reg\_param** | 0.0 (unregularized — full-rank on shared split) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support (Test) |
|---|---|---|---|---|
| **Adware** | 86.71% | 98.41% | **92.19%** | 126 |
| **Banking** | 84.00% | 90.00% | **86.90%** | 210 |
| **Benign** | 90.32% | 93.85% | **92.05%** | 179 |
| **Riskware** | 94.12% | 87.84% | **90.87%** | 255 |
| **SMS Malware** | 100.00% | 94.37% | **97.11%** | 391 |
| **Macro Average** | **91.03%** | **92.90%** | **91.82%** | **1,161** |
| **Weighted Average** | — | — | **92.58%** | **1,161** |

### Confusion Matrix (Official Test Split)

Rows = Ground Truth, Columns = Predicted:
```text
               Adware  Banking  Benign  Riskware  SMS Malware
Adware            124        0       1         1            0
Banking             8      189       8         5            0
Benign              0        8     168         3            0
Riskware           10       13       8       224            0
SMS Malware         1       15       1         5          369
```

### Regularisation Tuning (Validation Split Only)

| reg\_param | Val Macro F1 | Status |
|---|---|---|
| **0.00** | **0.9292** | ✅ PASS — **Selected** |
| 0.10 | 0.9186 | PASS |
| 0.25 | 0.8979 | PASS |
| 0.50 | 0.8760 | PASS |
| 0.75 | 0.8577 | PASS |
| 1.00 | 0.7427 | PASS |

> Unlike the 80:20 ablation (where SMS Malware's covariance is rank-deficient), the official 3-way split has sufficient training data per class for unregularized QDA to succeed.

### Configuration

Defined in [`configs/qda_static_k50.yaml`](configs/qda_static_k50.yaml):
- Feature table: `runs/real/artifacts/stage05_static_features.parquet`
- Split manifest: `runs/real/manifests/maldroid_split_manifest.parquet`
- Feature selection: `runs/real/artifacts/stage08_feature_selection.json`, `mutual_info`, K=50
- Regularisation candidates: `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]` (val-only selection)
- Output: `runs/group1_qda/official/`

### Running the Official Pipeline

```powershell
# 1. Validate data contract
python -m groups.group1_qda.src.data --config groups/group1_qda/configs/qda_static_k50.yaml

# 2. Train (outputs to runs/group1_qda/official/<timestamp>/)
python -m groups.group1_qda.src.train_qda --config groups/group1_qda/configs/qda_static_k50.yaml

# 3. Evaluate on test split (exactly once, on the frozen model)
python -m groups.group1_qda.src.evaluate_qda \
    --config groups/group1_qda/configs/qda_static_k50.yaml \
    --model-dir runs/group1_qda/official/<run-directory>
```

### Official-Run Artifacts

Located at `runs/group1_qda/official/20261029T041546_853367Z/`:

| File | Description |
|---|---|
| `qda_model.joblib` | Frozen model (imputer, variance filter, scaler, QDA with reg_param=0.0) |
| `training_metadata.json` | Train + val metrics, runtime, reg_param search, package versions |
| `data_validation_report.json` | Dataset fingerprint and schema checks |
| `selected_features.json` | Top-50 selected feature names |
| `evaluations/20261029T041552_638114Z/evaluation_results.json` | Full test metrics |
| `evaluations/20261029T041552_638114Z/confusion_matrix.png` | Seaborn heatmap |
| `evaluations/20261029T041552_638114Z/test_predictions.csv` | Per-sample predictions |

---

## 2. Supplementary Ablation — 80:20 Stratified Split

> ⚠️ **Not used for cross-group comparison.** This experiment uses an independent stratified re-split of all 11,598 samples (80% train / 20% test), bypassing the shared split contract (Integration Guide Section 4). It is kept as a labeled secondary ablation for internal diagnostics only.

### Results from 80:20 Ablation (Not Cross-Group Comparable)

Evaluated on 2,320 held-out test samples:

| Metric | Value |
|---|---|
| **Test Accuracy** | 91.08% |
| **Balanced Accuracy** | 91.54% |
| **Macro F1-Score** | 90.47% |
| **Weighted F1-Score** | 91.16% |
| **Selected reg\_param** | 0.0 (CV Macro F1 = 0.9127 ± 0.0058) |

> Note: Results differ from earlier runs (which reported reg_param=0.1) because the parquet's `__scaled` columns are now loaded directly, providing already-normalized inputs to the 80:20 pipeline. The covariance singularity that previously occurred with reg_param=0.0 no longer manifests.

### Running the 80:20 Ablation

```powershell
python -m groups.group1_qda.src.run_80_20_experiment \
    --config groups/group1_qda/configs/qda_static_k50_80_20.yaml
```

Output: `runs/group1_qda/ablation_80_20/run_<TIMESTAMP>/`

---

## 3. Running Unit Tests

```powershell
python -m pytest groups/group1_qda/tests/ -v
# 23/23 passed
```

---

## 4. Guardrails and Data Contract

- Never writes outputs inside `runs/real/` or `runs/kronodroid/` (protected by `_new_run_directory` guard).
- All preprocessing (imputer, variance filter, scaler, feature selection) is fitted on **train only**.
- `reg_param` is selected using **validation split only**.
- Test split is evaluated **exactly once** on the final frozen model.
- Feature vectors never contain `sample_id`, `split`, `Class`, `label_canonical`, or `dup_group`.
- Only K ∈ {50, 100, 270} from `stage08_feature_selection.json` are valid feature counts.