# Group 1 — Quadratic Discriminant Analysis (QDA) Android Malware Detection

This module implements Quadratic Discriminant Analysis (QDA) for Android malware classification on the MalDroid-2020 dataset (5 classes: Adware, Banking, Benign, Riskware, SMS Malware). It contains both the official 3-way partition baseline and the complete **80:20 Stratified Experiment** pipeline.

---

## 1. 80:20 Stratified Experiment (Primary Research Implementation)

The 80:20 experiment solves the memory and partition challenges of Android static feature analysis with the following innovations:
- **Memory-Efficient Parquet Column Pruning**: Loads only the K=50 selected features + `sample_id` from the 63,165-column static feature Parquet table (`stage05_static_features.parquet`), reducing peak memory from **5.47 GiB to ~4.4 MB**.
- **Deterministic 80:20 Stratified Split**: Combines all 11,598 available samples into an 80% train (9,278 samples) and 20% test (2,320 samples) split with identical class proportions and zero sample leakage.
- **Leakage-Safe Preprocessing**: Median imputation and standard scaling are fitted exclusively on the 80% training set and then applied to the 20% test set.
- **Regularization Tuning via 5-Fold Stratified CV**: Compares `reg_param` candidates `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`. Identifies that unregularized QDA (`reg_param=0.0`) fails due to collinearity in the SMS Malware covariance matrix, and selects `reg_param=0.1` (CV Macro-F1 = 0.9076 +/- 0.0053).
- **Comprehensive Visualizations and Artifacts**: Generates raw and normalized confusion matrices, per-class F1 bar charts, regularization search curves, and class distribution plots.

### Configuration
Defined in [`configs/qda_static_k50_80_20.yaml`](configs/qda_static_k50_80_20.yaml):
- Feature table: `runs/real/artifacts/stage05_static_features.parquet`
- Manifest: `runs/real/manifests/maldroid_split_manifest.parquet`
- Feature selection: `runs/real/artifacts/stage08_feature_selection.json` (modality: `static`, method: `mutual_info`, K: 50)
- Splitting strategy: `stratified_80_20` (test_size: 0.2, random_seed: 42)
- Regularization candidates: `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`

### Executing the Experiment

Run from repository root:
```powershell
python -m groups.group1_qda.src.run_80_20_experiment --config groups/group1_qda/configs/qda_static_k50_80_20.yaml
```

### Running Unit Tests

Run the dedicated test suite verifying splitting, preprocessing leakage safety, regularization, and metric computation:
```powershell
python -m unittest groups/group1_qda/tests/test_80_20_experiment.py
```

---

## 2. Experimental Results (80:20 Stratified Test Partition)

Evaluated on the held-out 20% test set (**2,320 samples**):

### Overall Metrics
| Metric | Value |
|---|---|
| **Test Accuracy** | **90.47%** (0.9047) |
| **Balanced Accuracy** | **90.47%** (0.9047) |
| **Macro Precision** | **89.03%** (0.8903) |
| **Macro Recall** | **90.47%** (0.9047) |
| **Macro F1-Score** | **89.49%** (0.8949) |
| **Weighted Precision** | **90.85%** (0.9085) |
| **Weighted Recall** | **90.47%** (0.9047) |
| **Weighted F1-Score** | **90.43%** (0.9043) |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support (Test Samples) |
|---|---|---|---|---|
| **Adware** | 87.86% | 98.01% | **92.66%** | 251 |
| **Banking** | 81.19% | 87.38% | **84.17%** | 420 |
| **Benign** | 84.40% | 91.92% | **88.00%** | 359 |
| **Riskware** | 93.38% | 77.60% | **84.76%** | 509 |
| **SMS Malware** | 98.32% | 97.44% | **97.88%** | 781 |
| **Macro Average** | **89.03%** | **90.47%** | **89.49%** | **2,320** |
| **Weighted Average** | **90.85%** | **90.47%** | **90.43%** | **2,320** |

### Confusion Matrix
Rows = Ground Truth, Columns = Predicted:
```text
               Adware  Banking   Benign  Riskware  SMS Malware
Adware            246        1        1         3            0
Banking             9      367       18        18            8
Benign              1       23      330         5            0
Riskware           24       48       37       395            5
SMS Malware         0       13        5         2          761
```

### Regularization Tuning Analysis
- `reg_param = 0.00`: **FAILED** (`LinAlgError: The covariance matrix of class SMS Malware is not full rank. Collinear features present.`)
- `reg_param = 0.10`: **CV Macro F1 = 0.9076 +/- 0.0053** (Optimal)
- `reg_param = 0.25`: CV Macro F1 = 0.8797 +/- 0.0064
- `reg_param = 0.50`: CV Macro F1 = 0.8622 +/- 0.0108
- `reg_param = 0.75`: CV Macro F1 = 0.8456 +/- 0.0115
- `reg_param = 1.00`: CV Macro F1 = 0.7096 +/- 0.0075

Shrinking covariance towards the diagonal via `reg_param=0.10` successfully stabilizes inverse covariance estimation while retaining quadratic boundary flexibility.

---

## 3. Generated Artifacts

Every run creates a timestamped folder under `outputs/qda_80_20_experiment/run_<TIMESTAMP>/` containing:
- `qda_model.joblib`: Serialized trained model with imputer, scaler, feature names, and best `reg_param`.
- `experiment_split_manifest.parquet` & `.csv`: Full sample-by-sample record mapping each sample to its 80% train or 20% test assignment.
- `test_metrics.json`: Full metrics payload including classification report and confusion matrix.
- `per_class_metrics.csv`: Tabular per-class precision, recall, F1, and support.
- `selected_features.json`: Ordered list of the top 50 selected static features.
- `experiment_metadata.json`: Timestamp, random seed, package versions, input file SHA256 hashes, and execution environment.
- `plots/`:
  - `confusion_matrix.png` & `confusion_matrix_normalized.png`
  - `per_class_f1.png`
  - `reg_param_search.png`
  - `class_distribution.png`

---

## 4. Original 3-Way Split Baseline (Data Contract Mode)

The original pipeline uses [`configs/qda_static_k50.yaml`](configs/qda_static_k50.yaml) with fixed train/val/test splits:

```powershell
# 1. Validate data contract
python -m groups.group1_qda.src.data --config groups/group1_qda/configs/qda_static_k50.yaml

# 2. Train baseline model
python -m groups.group1_qda.src.train_qda --config groups/group1_qda/configs/qda_static_k50.yaml

# 3. Evaluate baseline model
python -m groups.group1_qda.src.evaluate_qda --config groups/group1_qda/configs/qda_static_k50.yaml --model-dir "<run-directory>"
```