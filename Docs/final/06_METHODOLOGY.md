# Methodology
## Android Malware Classification — Group 3

**Document version:** Final (post-execution)  
**Style:** Academic technical

---

## 1. Problem Definition

Android malware detection is framed as a **five-class supervised classification problem**. Given a set of static and dynamic features extracted from an Android application, the classifier must assign the application to one of five categories: Adware, Banking malware, SMS Malware, Riskware, or Benign.

The classification objective is to maximise **macro-averaged F1-score** across all five classes, which weights each class equally regardless of its frequency in the dataset. Accuracy and balanced accuracy are reported as supporting metrics.

---

## 2. Dataset Selection

Two publicly released Android malware datasets were selected:

**MalDroid2020** (primary development dataset): 11,598 Android applications spanning five classes, with both static analysis (APK-derived feature vectors) and dynamic analysis (sandbox-recorded syscall/Binder event frequencies) representations. MalDroid2020 provides matched static and dynamic representations for each application, enabling cross-modal feature fusion.

**AndMal2020** (external validation dataset): 357,805 static records and 53,439 dynamic records from a broader taxonomy. AndMal2020 was designated for external generalization analysis after model development on MalDroid2020 was complete.

AndMal2020 was **not** used during any stage of model development, hyperparameter selection, preprocessing design, or feature selection. It serves exclusively as an independent evaluation corpus.

---

## 3. Data Audit

Stage 01 performed a comprehensive file-level audit of all datasets:

- **File fingerprinting:** SHA-256 prefixes computed for all raw data files to establish reproducibility anchors
- **Structural validation:** Row and column counts verified against documented dataset specifications
- **Label validation:** Numeric labels mapped to class names; distributions tabulated
- **Missingness analysis:** Zero missing values in both dynamic feature files; 18,931 all-missing columns identified in the static representation
- **Duplicate detection:** 77 exact duplicate rows found in dynamic-139 space; 72 in dynamic-470 space
- **Type sampling:** Static columns sampled to determine feature type distribution (binary, continuous, string, etc.)

Key findings: MalDroid static, dynamic-139, and dynamic-470 files all contain exactly 11,598 rows with consistent per-class label distributions — empirically validating the row-order correspondence between files.

---

## 4. Data Repair and Provenance

The MalDroid2020 static feature file (`feature_vectors_static.csv`) required repair before use:

1. **Encoding correction:** The raw file required UTF-7 decoding and column-header realignment
2. **Label attachment:** The static file contains no per-sample identifier or label column. The Class label was attached via row-order alignment with the dynamic feature file, which shares the same 11,598-sample population in the same row order
3. **Validation:** Alignment was empirically confirmed by matching row counts and per-class distributions across all three modalities
4. **Provenance recording:** The repair process, its assumptions, and limitations were recorded in the audit artifact (`stage01_data_audit.json`)

**Limitation:** Row-order alignment cannot be cryptographically verified because the static file carries no per-row SHA-256 hash or APK package identifier. This limitation is disclosed in all results.

The repaired file is saved as a Parquet artifact with SHA-256 prefix `fb5ce9ec`.

---

## 5. Label Handling

MalDroid2020 uses integer labels (1–5), which were mapped to class names:

| Integer | Class |
|:---:|---|
| 1 | Adware |
| 2 | Banking |
| 3 | SMS Malware |
| 4 | Riskware |
| 5 | Benign |

For AndMal2020, the more granular taxonomy was mapped to the same five classes where possible. Categories without a corresponding MalDroid class (Backdoor, Ransomware, Trojan, etc.) were excluded from the 5-class evaluation subset.

---

## 6. Feature Typing

Stage 03 inspected all 50,620 columns of the MalDroid static representation to determine which columns are suitable for machine learning:

| Type | Count | Decision | Rationale |
|---|:---:|---|---|
| Binary numeric | 31,317 | Usable | Valid feature signal |
| Count/continuous | 335 | Usable | Valid feature signal |
| All-missing | 18,931 | Dropped | No information content |
| Zero-variance | 22 | Dropped | No discriminative power |
| High-cardinality string | 4 | Dropped | Not suitable for direct numeric input |
| Categorical string | 11 | Dropped | Not encoded for tree models |
| Identity/row-index | 1 | Dropped | Not a feature |

**Usable total:** 31,652 columns.

---

## 7. Leakage-Safe Splitting

Stage 04 implements a stratified train/validation/test split with the following properties:

- **Stratification:** Split proportions are maintained within each class to preserve the original class distribution in all three splits
- **Determinism:** Global seed 42 ensures exact reproducibility
- **Duplicate-group handling:** Rows identified as exact duplicates in the 470-feature dynamic space are grouped. Each group is assigned entirely to one split, preventing any case where the same behavioral fingerprint appears in both train and test
- **Cross-class conflict resolution:** 3 groups of samples share identical dynamic feature vectors across different class labels. Each group is assigned to one split in its entirety, with the resolved split recorded in the audit artifact

Final counts: Train = 9,281; Validation = 1,156; Test = 1,161.

---

## 8. Static Preprocessing

The 31,652 usable static columns are preprocessed using `sklearn.preprocessing.StandardScaler`:

- **Fitting:** The scaler is fitted **exclusively on training rows** (9,281 samples)
- **Transformation:** The fitted scaler is applied (transform only, no refit) to validation and test rows
- **Persistence:** The fitted scaler is serialized as `stage05_static_scaler.joblib` and used consistently in all downstream transformations
- **No test leakage:** No statistics from validation or test sets inform the scaler parameters

The scaled output is stored in `stage05_static_features.parquet` (11,598 × 31,652). Both raw and scaled views are available in the fused artifact for compatibility with downstream model families that require normalized inputs.

---

## 9. Dynamic Preprocessing

Two dynamic feature spaces are preprocessed independently:

**Dynamic-139** (syscall frequency only, 139 features) and **Dynamic-470** (syscall + Binder/composite, 470 features) are each processed with `StandardScaler` fitted on training rows only. Both scalers are persisted as joblib artifacts.

Dynamic features had zero missing values and zero zero-variance columns — no additional cleaning was required beyond scaling.

---

## 10. Feature Selection

Feature selection reduces the high-dimensional feature space to a compact, discriminative subset:

**Methods:**
- **Mutual information** (`sklearn.feature_selection.mutual_info_classif`): Measures information-theoretic dependence between each feature and the label
- **Chi-squared** (`sklearn.feature_selection.chi2`): Measures statistical dependence for non-negative features
- **LightGBM importance** (`lightgbm.LGBMClassifier.feature_importances_`): Tree-based split-gain importance scores

**Protocol:**
- All selectors fitted on **training data only**
- K values evaluated: 50, 100, 270
- All four modalities evaluated independently
- Final selection determined by downstream model performance on the validation set

**Final configuration:** Modality = fused; K = 270. The 270 selected features comprise 152 static and 118 dynamic-470 origin features.

---

## 11. Modality Construction

Four modality configurations are evaluated:

| Modality | Construction | Features |
|---|---|:---:|
| static | Static branch only (selected K columns) | 270 |
| dynamic_139 | Dynamic-139 branch only (selected K columns) | 139 |
| dynamic_470 | Dynamic-470 branch only (selected K columns) | 270 |
| fused | Static + Dynamic-470 columns concatenated row-wise | 270 |

Fusion is valid for MalDroid2020 because the static and dynamic files share the same population in the same row order. Fusion produces a complementary feature set combining static structural analysis with dynamic behavioral observations.

---

## 12. LightGBM Training

LightGBM (`lightgbm==4.7.0`) is trained with the following configuration:

- `n_estimators=300`, `num_leaves=31`, `learning_rate=0.05`, `device_type='cpu'`
- `class_weight` set to inverse-frequency weights computed from the training set
- `random_state=42` for reproducibility

Training was performed exclusively on training split samples. The validation split was used for modality selection (Stage 13 ablation). The test split was held out until final evaluation.

**GPU note:** The installed LightGBM wheel does not include GPU Tree Learner support. All training ran on CPU (Intel, 16 physical cores). Training time for the final locked model: 11.49 seconds.

---

## 13. Evaluation Methodology

**Internal evaluation:**

- Performance measured on the held-out **test split** (1,161 samples, never seen during training or selection)
- Metrics: Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy, per-class Precision/Recall/F1
- Macro-F1 used as the primary selection criterion (equal class weighting)
- Confusion matrix recorded for each modality

**Stage 14 verification:**

The final locked model is retrained on the combined train+validation set and evaluated on the test set. Metrics are verified to match Stage 13 within a tolerance of 0.01. `metrics_match_stage13 = True` confirmed.

---

## 14. External Validation

Stage 15 attempted to evaluate the locked fused model on AndMal2020's dynamic data by matching the required 270 feature names against AndMal's 144 dynamic column names.

**Result:** 0 of 270 required feature names were found in AndMal's dynamic representation.

**Cause:** MalDroid's dynamic features are syscall/Binder event frequencies; AndMal's are memory/API/battery/logcat metrics. Different instrumentation frameworks produce entirely disjoint feature name spaces.

**Decision:** Zero-filling missing features was rejected as scientifically invalid (treating "not measured" as "measured and zero"). The result is reported as a feature-space incompatibility, not a model failure.

---

## 15. Explainability

The frozen model supports post-hoc explainability via:

- **LightGBM native importance:** Split-gain and split-count importances available directly from the model object
- **SHAP TreeExplainer:** SHAP 0.49.1 is installed and supports `TreeExplainer` for LightGBM without retraining

SHAP analysis was not executed as part of the `runs/real` pipeline. SHAP provides model attribution — characterizing which features the model relied on for given predictions. SHAP values are not causal evidence.

---

## 16. Reproducibility

The pipeline is designed for full reproducibility:

- Global seed 42 propagated to all stochastic operations
- All preprocessing artifacts (scalers, selectors) persisted as joblib/JSON artifacts
- Configuration snapshot saved at run time (`config.yaml.snapshot.json`)
- Environment snapshot saved at run time (`environment.json`)
- Stage completion tracked via config-hash-based resumability (`status.json`)
- Frozen model artifact checksummed (SHA-256 prefix: `ab63ed065259b79e`)

To reproduce: run `python one_run.py` from the project root with `config.yaml`. Previously completed stages are skipped automatically based on config-hash matching.

---

## 17. Limitations

1. **Static alignment provenance:** Row-order alignment is empirically validated but not cryptographically verifiable
2. **Cross-class duplicates:** 3 duplicate groups with contradictory labels represent an inherent performance ceiling for the dynamic feature representation
3. **GPU unavailability:** LightGBM CPU-only; no GPU speed benefit was measured
4. **External validation:** Not executable for the locked fused model due to feature-space incompatibility
5. **Single primary dataset:** All model development decisions were informed by MalDroid2020 alone; generalizability to other populations is unverified
6. **Fixed taxonomy:** The 5-class taxonomy may not represent the full diversity of the malware landscape; unmappable AndMal categories (Ransomware, Trojan, Backdoor, etc.) were excluded
7. **Temporal scope:** Results are bounded to the malware families and analysis period represented in MalDroid2020

---

*This methodology document describes what was actually implemented and executed in `runs/real`. No claims are made beyond the scope of the conducted experiments.*
