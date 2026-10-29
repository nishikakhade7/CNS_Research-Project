# Group 1 KronoDroid QDA plan

## Scope and verified inputs

Build a separate binary classifier: Benign = 0, Malware = 1, real-device apps only.
Leave all existing MalDroid files unchanged and consume Group 3 artifacts read-only.
Never write inside `runs/kronodroid/` or `runs/real/`.

Inputs (repository-relative):

- `runs/kronodroid/artifacts/kd05_features_scaled.parquet`: 78,129 rows, 475 numeric features, plus `split` and `label_binary`. SHA256 values are in the unique, unnamed index (`index.name is None`), matching the manifest IDs.
- `runs/kronodroid/manifests/kd04_split_manifest.parquet`: authoritative IDs, labels and split assignments; cross-check against kd05.
- `runs/kronodroid/artifacts/kd06_feature_rankings.json`: train-only rankings and ordered top-K lists supplied by Group 3.
- `runs/kronodroid/metrics/kd09_eval_metrics.json`: Group 3 comparison results only.

Preserve the exact splits: train 54,691 (25,726 benign / 28,965 malware), val 11,720 (5,513 / 6,207), test 11,718 (5,512 / 6,206). Do not re-split or use the MalDroid 80:20 ablation.

## Separate implementation and configuration

Add `configs/qda_kronodroid.yaml` and `src/run_qda_kronodroid.py` under this directory. Do not adapt or import MalDroid's `data.py`: its `__scaled` naming and feature-list layout differ.

Put experiment settings, paths, expected data contract, feature-set order, seed (42), output settings and reporting choices in YAML; require configuration rather than silently substituting experiment defaults. QDA itself is deterministic and has no `random_state` argument.

Validate inputs before fitting: dimensions, labels, split counts, ID uniqueness and split disjointness, manifest alignment, numeric finite features, and feature-list membership/order. Stop on a mismatch. Record input SHA256 hashes and exact installed package versions.

## Training and validation selection

1. Use the ordered Group 3 lists: primary `mi_top50`; comparisons `mi_top30`, `mi_top100`, `et_top50`. The latter uses the replication model's feature identities; QDA will still consume kd05's standardized values, so its representation differs from Group 3's MinMax replication model.
2. Set X to selected feature columns and y to `label_binary`. Exclude split, labels and identifiers. Do not compute a new ranking, scale again, impute, resample or apply class weights. Drop only features with exactly zero variance in TRAIN and record every dropped name. Preserve the remaining list order.
3. For each feature set, fit sklearn `QuadraticDiscriminantAnalysis` candidates on TRAIN only with `reg_param` grid `[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]`. Leave `priors` and `tol` at sklearn defaults and record their installed-version values.
4. Score candidates on VAL macro F1; record VAL accuracy, fit duration, and every warning with category and message. On an exact macro-F1 tie, choose the larger `reg_param`. Preserve duplicate warning occurrences. Record a candidate failure and stop rather than silently changing the experiment.
5. Retain each winning fitted TRAIN model without refitting. Select the overall feature set by VAL macro F1 only. For an exact tie across feature sets, use the configured order (`mi_top50`, `mi_top30`, `mi_top100`, `et_top50`), keeping the primary first. This cross-feature-set tie rule is an implementation assumption, declared before testing.
6. Freeze all four winners and the overall selection before any TEST prediction. Evaluate each feature set's winner on TEST exactly once; reuse its saved predictions/probabilities for metrics and plots. Do not change the experiment after seeing TEST results.

## Outputs

Use `runs/group1_qda/kronodroid/<UTC-timestamp>/`, refusing existing run directories and resolving paths before checking both protected Group 3 directories. Save a root selection summary and provenance, plus one subdirectory per feature set to avoid ambiguous models or metrics.

For each feature set save `model.joblib` (model, retained feature list, dropped features, classes), `config_used.yaml`, `package_versions.json`, `random_seed` (42), `reg_param_search.json`, `train_metrics.json`, `val_metrics.json`, `test_metrics.json`, `test_predictions.csv`, `confusion_matrix.png`, and `runtime.json`. Capture any evaluation warnings separately as well.

Metrics: accuracy, balanced accuracy, Malware precision/recall/F1 (positive label 1), macro F1, ROC-AUC from the class-1 `predict_proba` column, and confusion matrix ordered [0, 1]. Predictions: SHA256 index exported as `sha256`, true label, predicted label, probability of Malware. Configure and disclose undefined-metric handling.

Runtime must distinguish the retained winning candidate's fit time alone from the full feature-set search time; record overall search time separately. Never label the full search as final-model training time.

## Verification, reporting and delivery

Add the requested tests for split disjointness, metadata exclusion, feature-list membership and output-path protection; also cover the new path guard's `runs/real/` protection and the selection tie rules. Run pytest and report its actual output. Do not execute TEST evaluation from tests.

Write a new KronoDroid README with provenance hashes, counts, every selected setting and its chooser/reason, explicit assumptions, exact JSON metrics side by side, all warnings and their meaning, and VERIFIED / NOT VERIFIED claim labels. Attribute Group 3's comparison numbers exactly: replication accuracy 0.9779 / F1 0.979; full_standardized accuracy 0.9834 / F1 0.9842. Their stored metrics have limited precision; do not invent additional digits or assume their F1 is macro F1. Describe representation differences and binary/real-device/dataset limitations. Audit the README against outputs and correct inconsistencies.

Proceed one user-approved step at a time. Before delivery, show git status and the exact proposed files. Force-add prediction CSVs when staging because of `.gitignore`; ask before committing any `model.joblib`. Do not commit or push without the requested approval. Push only `grp1-qda-kronodroid` to `origin` at `nishikakhade7/CNS_Research-Project`, never master or DeepMehta06.
