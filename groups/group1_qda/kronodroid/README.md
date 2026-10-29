# Group 1: KronoDroid binary QDA

**VERIFIED** - Final run: `runs/group1_qda/kronodroid/20261029T080353_215967Z`. Validation selected **et_top50, reg_param=0.1**. Its TEST accuracy is **0.911247653183137**, Malware F1 **0.9128832300217792**, and macro F1 **0.9112163584445303**. These are QDA results; they are below Group 3's recorded LightGBM accuracy and Malware F1. The prespecified primary experiment remains `mi_top50`; the overall validation winner is reported separately.

Claim labels apply to the paragraph, list, code block or table immediately following them. VERIFIED means checked against the named local files, implementation or execution output; it does not mean independently validated against raw apps. NOT VERIFIED identifies untested assumptions or claims. Numbers below use the values stored in JSON without display rounding or invented digits.

## Scope and input provenance

**VERIFIED** - This new module consumes Group 3 artifacts without modifying their pipeline. It does not import or alter the existing five-class MalDroid implementation. The task is Benign (0) versus Malware (1), `kronodroid_real_device` only. All paths in this document are repository-relative unless stated otherwise.

**VERIFIED** - Input digests from `input_provenance.json`, independently rechecked while writing this report:

| Input | Path | SHA256 |
| --- | --- | --- |
| features | `runs/kronodroid/artifacts/kd05_features_scaled.parquet` | `dc55adc63f0ae38d6cc5728c69eaf2699c1ab17fa8e1367bcbbcff4a80f0386c` |
| manifest | `runs/kronodroid/manifests/kd04_split_manifest.parquet` | `9f4047a3fad0188760b105e60bc31f4d46c4cc16489bc7c106ff903c262ba219` |
| rankings | `runs/kronodroid/artifacts/kd06_feature_rankings.json` | `d70a46502b422dd36ac44fc29017fe3a8499d92b3ae4abfa7de6818275b547e5` |
| group3_metrics | `runs/kronodroid/metrics/kd09_eval_metrics.json` | `bcdb7a3053978f10d8dc3b1583483187c37160b3d61da82fdd24dc489335743c` |

**VERIFIED** - kd05 has 78,129 rows and 477 columns: 475 numeric features plus `split` and `label_binary`. SHA256 IDs are the unique, unnamed index (`index.name is None`), not a feature column. IDs, splits and binary labels match kd04. Feature values contain no missing or infinite entries. Neither duplicates nor IDs spanning splits were found. Each ranking is a complete ordering of 475 unique features; all six top-K lists match ranking prefixes and exist in kd05.

**VERIFIED** - Split and label counts, cross-checked from the actual Parquets (not newly generated splits):

| Split | Benign (0) | Malware (1) | Total |
| --- | --- | --- | --- |
| train | 25726 | 28965 | 54691 |
| val | 5513 | 6207 | 11720 |
| test | 5512 | 6206 | 11718 |
| Total | 36751 | 41378 | 78129 |

**VERIFIED** - `pipeline/kronodroid/stage05_preprocess.py` applies log1p to nonnegative dynamic count-like columns and fits StandardScaler on TRAIN only. It fills missing upstream values with zero before preprocessing; Group 1 does not repeat that operation. `stage06_selection.py` fits its MinMax selection representation, ExtraTrees and mutual information on TRAIN only. Group 1 consumes their existing ordered lists and standardized kd05 values without re-ranking or re-scaling. These preprocessing claims are verified from source and consistency checks, not by independently rebuilding Group 3 artifacts from raw apps.

## Every stage and methodological choice

**VERIFIED** - Chooser attribution: User = original task or explicit subsequent approval; Group 3 = supplied pipeline and artifacts; sklearn = installed library defaults; Implementation = routine implementation choice documented in PLAN.md or configuration before TEST evaluation.

| Stage / choice | Value or behavior | Who chose it | Why |
| --- | --- | --- | --- |
| Setup | Separate branch grp1-qda-kronodroid; separate module under groups/group1_qda/kronodroid | User | Preserve the existing MalDroid work and Group 3 artifacts |
| Environment | Python 3.14.3; sklearn 1.7.2; full versions below | Implementation, user approved version change | Use Group 3 documented sklearn version after first run failed; selected before any TEST prediction |
| Data contract | 78,129 rows; 475 features; unnamed SHA256 index | Group 3; verified by Group 1 | Fail on unexpected artifacts rather than silently adapt |
| Labels | Benign=0, Malware=1 | User / Group 3 | Binary real-device detection only |
| Split | TRAIN 54,691 / VAL 11,720 / TEST 11,718; no re-split | Group 3; user required reuse | Keep assignments and holdouts fixed |
| Feature sets | mi_top50 primary; mi_top30, mi_top100, et_top50 comparisons | User | Prespecified K and ranking methods, not selected using TEST |
| Preprocessing | Use kd05 directly; no added scaling, imputation, resampling or class weighting | User / implementation | Consume existing train-only standardized data; QDA uses default empirical priors |
| Constant filter | TRAIN variance == 0.0, ddof=0; preserve retained order | User; implementation chose variance definition | Drop only globally constant TRAIN features; none dropped in the four fitted sets |
| Estimator | sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis | User | QDA only |
| Regularisation grid | [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] | User | Prespecified regularisation search |
| Installed QDA defaults before grid search | {"priors": null, "reg_param": 0.0, "store_covariance": false, "tol": 0.0001} | sklearn defaults, user required priors/tol defaults | Only reg_param is overridden, using the configured grid; other constructor settings remain default |
| Priors | priors=None; fitted proportions [0.47038818087071, 0.52961181912929] | sklearn / TRAIN counts | Empirical class proportions; not balanced priors |
| Random seed | 42; QDA has no random_state argument | User | Record seed; no new stochastic ranking or split is generated |
| Candidate selection | Largest VAL macro F1; exact ties choose larger reg_param | User | Use validation only; prefer greater regularisation on a tie |
| Feature-set selection | Largest VAL macro F1; exact ties use mi_top50, mi_top30, mi_top100, et_top50 order | User chose VAL selection; implementation chose cross-set tie order | Declare deterministic tie order before TEST, keeping primary first |
| Failures and warnings | Capture every warning occurrence; record candidate failure and stop | User stop rule / PLAN.md | Do not hide rank deficiency or silently skip failed candidates |
| Freeze | Keep winning candidate fitted on TRAIN only; no train+val refit | User | Retain the model selected on VAL |
| TEST | One evaluation per frozen feature-set winner, after all selections | User | Use one predict and one predict_proba call per winner; reuse arrays for all metrics and plots |
| Metrics | Accuracy, balanced accuracy, Malware precision/recall/F1, macro F1, ROC-AUC, confusion matrix | User | Report class-specific and overall performance |
| Metric conventions | Positive label 1; confusion order [0,1]; zero_division=0; AUC uses class-1 probability | User / implementation | Explicit label orientation and undefined-metric behavior |
| Output guard | Resolve paths; reject runs/kronodroid and runs/real and descendants; constrain output to runs/group1_qda/kronodroid | User / implementation | Protect Group 3 and avoid overwriting existing runs |
| Output layout | UTC timestamp root; root selection/provenance; one subdirectory per feature set | User / implementation | Keep four sets of models and metrics unambiguous |
| Runtime | Winning candidate fit call alone; feature-set search separately; outer all-set search separately | User / implementation | Do not confuse search wall time with final-model training time |
| Verification | 26 pytest cases passed in 3.75s on Python 3.14.3 / pytest 9.2.0 | User / implementation | Check leakage contracts, feature exclusions, guards and ties without model execution |
| Write-up | Exact JSON values, warning disclosure, assumptions and critical audit | User | Make limitations and provenance reviewable |
| Delivery | No commit or push yet; request approval, including model.joblib separately | User | Push only approved branch to nishikakhade7/CNS_Research-Project |

**VERIFIED** - The following complete configuration inventory supplies every YAML value, including reporting settings and filenames. Contract values are Group 3 facts checked by Group 1; input paths point to their supplied files. The selection/grid/seed choices above are user-specified except the declared cross-set tie rule. Reporting mechanics are implementation choices. The source YAML is `groups/group1_qda/kronodroid/configs/qda_kronodroid.yaml`; the executed snapshot is the final run's `config_used.yaml`. Fixed protected paths are deliberate code safety invariants rather than configurable escape routes.

| Config key | Exact value | Chooser | Purpose |
| --- | --- | --- | --- |
| `random_seed` | `42` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `inputs.features` | `"runs/kronodroid/artifacts/kd05_features_scaled.parquet"` | User / Group 3 | Read supplied artifacts; do not rebuild them |
| `inputs.manifest` | `"runs/kronodroid/manifests/kd04_split_manifest.parquet"` | User / Group 3 | Read supplied artifacts; do not rebuild them |
| `inputs.rankings` | `"runs/kronodroid/artifacts/kd06_feature_rankings.json"` | User / Group 3 | Read supplied artifacts; do not rebuild them |
| `inputs.group3_metrics` | `"runs/kronodroid/metrics/kd09_eval_metrics.json"` | User / Group 3 | Read supplied artifacts; do not rebuild them |
| `contract.rows` | `78129` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.feature_count` | `475` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.index_name` | `null` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.id_pattern` | `"^[0-9a-fA-F]{64}$"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_column` | `"split"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_column` | `"label_binary"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.manifest_id` | `"sample_id"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.manifest_label` | `"label_binary_raw"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.source_column` | `"source_dataset"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.source_value` | `"kronodroid_real_device"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.canonical_label_column` | `"label_binary_canonical"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.labels.0` | `"Benign"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.labels.1` | `"Malware"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.forbidden_features` | `["split", "label_binary", "sha256", "sample_id", "label_binary_raw", "label_binary_canonical", "family_label", "MalFamily_raw", "Package", "source_dataset", "Class", "label_canonical", "dup_group"]` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.forbidden_suffix` | `"__scaled"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_names.train` | `"train"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_names.val` | `"val"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_names.test` | `"test"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_counts.train` | `54691` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_counts.val` | `11720` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.split_counts.test` | `11718` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.train.0` | `25726` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.train.1` | `28965` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.val.0` | `5513` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.val.1` | `6207` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.test.0` | `5512` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.label_counts.test.1` | `6206` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.ranking_methods` | `["mutual_info", "extra_trees_importance"]` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top30.method` | `"mutual_info"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top30.k` | `30` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top50.method` | `"mutual_info"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top50.k` | `50` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top100.method` | `"mutual_info"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.mi_top100.k` | `100` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top30.method` | `"extra_trees_importance"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top30.k` | `30` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top50.method` | `"extra_trees_importance"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top50.k` | `50` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top100.method` | `"extra_trees_importance"` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `contract.top_k_sets.et_top100.k` | `100` | Group 3 facts / implementation validation | Enforce the verified input schema, identity and labels |
| `selection.feature_sets` | `["mi_top50", "mi_top30", "mi_top100", "et_top50"]` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `selection.primary` | `"mi_top50"` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `selection.metric` | `"macro_f1"` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `selection.candidate_tie` | `"larger_reg_param"` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `selection.feature_set_tie` | `"configured_order"` | Implementation, declared in plan | Deterministic exact cross-set ties |
| `selection.zero_variance` | `0.0` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `selection.variance_ddof` | `0` | Implementation | Population variance for exact constant filtering |
| `qda.reg_param_grid` | `[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `qda.expected_defaults.priors` | `null` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `qda.expected_defaults.tol` | `0.0001` | User; sklearn supplies asserted defaults | Prespecified experiment and documented default checks |
| `metrics.positive_label` | `1` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `metrics.confusion_labels` | `[0, 1]` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `metrics.zero_division` | `0` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `metrics.probability_atol` | `1e-12` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `metrics.probability_rtol` | `1e-12` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.directory` | `"runs/group1_qda/kronodroid"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.timestamp_format` | `"%Y%m%dT%H%M%S_%fZ"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.json_indent` | `2` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.model` | `"model.joblib"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.config` | `"config_used.yaml"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.versions` | `"package_versions.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.seed` | `"random_seed"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.search` | `"reg_param_search.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.train` | `"train_metrics.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.val` | `"val_metrics.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.test` | `"test_metrics.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.predictions` | `"test_predictions.csv"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.confusion` | `"confusion_matrix.png"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.runtime` | `"runtime.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.warnings` | `"evaluation_warnings.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.provenance` | `"input_provenance.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.selection` | `"selection_summary.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.filenames.status` | `"run_status.json"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.prediction_columns.id` | `"sha256"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.prediction_columns.truth` | `"true_label"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.prediction_columns.prediction` | `"predicted_label"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `output.prediction_columns.probability` | `"prob_malware"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.backend` | `"Agg"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.figsize` | `[6, 5]` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.dpi` | `150` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.cmap` | `"Blues"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.format` | `"d"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.title` | `"KronoDroid QDA test confusion matrix"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.xlabel` | `"Predicted label"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |
| `plot.ylabel` | `"True label"` | Implementation within user output requirements | Explicit metrics, file layout, serialization or plotting |

## Environment and failed first attempt

**VERIFIED** - The first attempt at `runs/group1_qda/kronodroid/20261029T075104_307506Z` used scikit-learn 1.9.1 and failed on `mi_top50`, reg_param=0.0 with ``LinAlgError: The covariance matrix of class 0 is not full rank. Increase the value of `reg_param` to reduce the collinearity.`` Its search JSON records FAIL with no warnings; its run status records no TEST evaluations started or completed. No candidate was silently skipped. The user approved changing versions, and a separate `.venv-sklearn172` environment was created before the successful run. The original `.venv` and failure artifacts were retained. The successful run uses the same experiment YAML and runner; model settings and splits were not changed. Only sklearn and its resolved dependency set differ between the environments. This matches Group 3's documented sklearn version, not their entire environment.

**VERIFIED** - Successful-run versions from `package_versions.json`; `pip check` reported `No broken requirements found.` Python: **3.14.3**.

| Package | Exact version |
| --- | --- |
| cloudpickle | 3.1.2 |
| colorama | 0.4.6 |
| contourpy | 1.4.0 |
| cycler | 0.12.1 |
| fonttools | 4.68.0 |
| iniconfig | 2.3.1 |
| joblib | 1.6.0 |
| kiwisolver | 1.5.1 |
| matplotlib | 3.11.2 |
| numpy | 2.5.4 |
| packaging | 26.3 |
| pandas | 3.1.0 |
| pillow | 13.0.0 |
| pip | 25.3 |
| pluggy | 1.7.0 |
| pyarrow | 26.0.0 |
| Pygments | 2.21.0 |
| pyparsing | 3.3.3 |
| pytest | 9.2.0 |
| python-dateutil | 3.0.0 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.18.1 |
| seaborn | 0.13.2 |
| six | 1.17.0 |
| threadpoolctl | 3.7.0 |
| tzdata | 2026.6 |

**VERIFIED** - The environment's local `requirements.lock.txt` and version JSON also reside inside `.venv-sklearn172`. Virtual environments are local setup artifacts. Portable version provenance is saved in the run's package JSON. The runner records versions but does not enforce sklearn 1.7.2 itself; use the pinned environment when reproducing.

## Selection results and full validation search

**VERIFIED** - Values below are copied from `selection_summary.json` and each set's `reg_param_search.json`. No model was refit after candidate selection; there were no exact winning-score ties. Dropped-feature lists are empty for every fitted feature set.

| Feature set | Requested K | Retained K | Dropped names | Chosen reg_param | VAL macro F1 |
| --- | --- | --- | --- | --- | --- |
| mi_top50 | 50 | 50 | [] | 0.5 | 0.8953924153109059 |
| mi_top30 | 30 | 30 | [] | 0.75 | 0.8854936767818297 |
| mi_top100 | 100 | 100 | [] | 0.01 | 0.9004263014711802 |
| et_top50 | 50 | 50 | [] | 0.1 | 0.9083442627959617 |

**VERIFIED** - All 32 candidate search records, with no rounding. Fit warnings are counted here and reproduced below:

| Set | reg_param | VAL macro F1 | VAL accuracy | Status | Warning count |
| --- | --- | --- | --- | --- | --- |
| mi_top50 | 0.0 | 0.8732195940260367 | 0.8732935153583617 | PASS | 2 |
| mi_top50 | 0.01 | 0.8782028435741736 | 0.8782423208191126 | PASS | 0 |
| mi_top50 | 0.05 | 0.8814504633120189 | 0.8814846416382253 | PASS | 0 |
| mi_top50 | 0.1 | 0.8809256337561517 | 0.8809726962457338 | PASS | 0 |
| mi_top50 | 0.25 | 0.8849386222693192 | 0.8849829351535836 | PASS | 0 |
| mi_top50 | 0.5 | 0.8953924153109059 | 0.8953924914675768 | PASS | 0 |
| mi_top50 | 0.75 | 0.8912018449394541 | 0.8912116040955631 | PASS | 0 |
| mi_top50 | 1.0 | 0.7812589814587642 | 0.7819112627986348 | PASS | 0 |
| mi_top30 | 0.0 | 0.8490574236957346 | 0.8494880546075085 | PASS | 2 |
| mi_top30 | 0.01 | 0.8549001568402651 | 0.8552047781569966 | PASS | 0 |
| mi_top30 | 0.05 | 0.8565236026164815 | 0.8568259385665529 | PASS | 0 |
| mi_top30 | 0.1 | 0.8582314608643378 | 0.8585324232081911 | PASS | 0 |
| mi_top30 | 0.25 | 0.8674032478996144 | 0.8675767918088737 | PASS | 0 |
| mi_top30 | 0.5 | 0.8813847234812493 | 0.8813993174061433 | PASS | 0 |
| mi_top30 | 0.75 | 0.8854936767818297 | 0.8854948805460751 | PASS | 0 |
| mi_top30 | 1.0 | 0.7957121474693898 | 0.796160409556314 | PASS | 0 |
| mi_top100 | 0.0 | 0.8970094342833627 | 0.8970136518771331 | PASS | 2 |
| mi_top100 | 0.01 | 0.9004263014711802 | 0.9004266211604095 | PASS | 0 |
| mi_top100 | 0.05 | 0.8988887172419504 | 0.8988907849829352 | PASS | 0 |
| mi_top100 | 0.1 | 0.8981997059403233 | 0.8982081911262798 | PASS | 0 |
| mi_top100 | 0.25 | 0.8952692506942989 | 0.8953071672354949 | PASS | 0 |
| mi_top100 | 0.5 | 0.8963729271788656 | 0.8964163822525597 | PASS | 0 |
| mi_top100 | 0.75 | 0.896407585543205 | 0.8964163822525597 | PASS | 0 |
| mi_top100 | 1.0 | 0.7833489690859436 | 0.7842150170648464 | PASS | 0 |
| et_top50 | 0.0 | 0.9059784263459237 | 0.9060580204778157 | PASS | 2 |
| et_top50 | 0.01 | 0.9074420942638093 | 0.9075085324232082 | PASS | 0 |
| et_top50 | 0.05 | 0.9083294734542277 | 0.9083617747440274 | PASS | 0 |
| et_top50 | 0.1 | 0.9083442627959617 | 0.9083617747440274 | PASS | 0 |
| et_top50 | 0.25 | 0.9063133793203866 | 0.9063139931740615 | PASS | 0 |
| et_top50 | 0.5 | 0.9071665434261063 | 0.9071672354948805 | PASS | 0 |
| et_top50 | 0.75 | 0.9042488094450648 | 0.9042662116040956 | PASS | 0 |
| et_top50 | 1.0 | 0.8151281910515558 | 0.8156143344709897 | PASS | 0 |

## Exact TRAIN / VAL / TEST results

**VERIFIED** - Each table is copied from that feature set's `train_metrics.json`, `val_metrics.json` and `test_metrics.json`. TRAIN metrics are in-sample, not a generalisation estimate. Macro F1 averages both classes; Malware F1 refers to class 1 only. ROC-AUC uses the class-1 predict_proba column, not hard labels. Confusion rows are true labels, columns predicted labels, both ordered [Benign (0), Malware (1)].

### mi_top50

**VERIFIED** - Saved metrics:

| Metric | TRAIN | VAL | TEST |
| --- | --- | --- | --- |
| accuracy | 0.8932365471466969 | 0.8953924914675768 | 0.8959720088752346 |
| balanced_accuracy | 0.8966891671286183 | 0.8985938621401073 | 0.8991308682500712 |
| malware_precision | 0.9544847103215156 | 0.9525713247319644 | 0.9524587189257848 |
| malware_recall | 0.838391161746936 | 0.8445303689382955 | 0.8457943925233645 |
| malware_f1 | 0.8926792508316944 | 0.8953031596925705 | 0.8959631304941538 |
| macro_f1 | 0.8932336681625502 | 0.8953924153109059 | 0.895972008117629 |
| roc_auc | 0.959649490516445 | 0.9591424005319121 | 0.9596160160563751 |
| confusion_matrix | [[24568, 1158], [4681, 24284]] | [[5252, 261], [965, 5242]] | [[5250, 262], [957, 5249]] |
| n_samples | 54691 | 11720 | 11718 |

### mi_top30

**VERIFIED** - Saved metrics:

| Metric | TRAIN | VAL | TEST |
| --- | --- | --- | --- |
| accuracy | 0.8840577060211003 | 0.8854948805460751 | 0.8857313534732889 |
| balanced_accuracy | 0.8872346001849096 | 0.888418051730095 | 0.8888034462178322 |
| malware_precision | 0.9407387204862464 | 0.9382093316519546 | 0.9407715993479442 |
| malware_recall | 0.833592266528569 | 0.8390526824552924 | 0.8369320012890751 |
| malware_f1 | 0.8839303692042979 | 0.88586494301752 | 0.8858190500554276 |
| macro_f1 | 0.8840575664764526 | 0.8854936767818297 | 0.8857312860662312 |
| roc_auc | 0.9499436827513641 | 0.9491804467265167 | 0.9494496699434556 |
| confusion_matrix | [[24205, 1521], [4820, 24145]] | [[5170, 343], [999, 5208]] | [[5185, 327], [1012, 5194]] |
| n_samples | 54691 | 11720 | 11718 |

### mi_top100

**VERIFIED** - Saved metrics:

| Metric | TRAIN | VAL | TEST |
| --- | --- | --- | --- |
| accuracy | 0.9037135909016109 | 0.9004266211604095 | 0.9025430960914832 |
| balanced_accuracy | 0.9067434184944342 | 0.9034885278263884 | 0.9055475657482084 |
| malware_precision | 0.9581287454088536 | 0.9553668232743043 | 0.9565452578434908 |
| malware_recall | 0.8555843259105818 | 0.851780248106976 | 0.8548179181437319 |
| malware_f1 | 0.9039576873974102 | 0.9006047185077932 | 0.9028250510551395 |
| macro_f1 | 0.9037129689370098 | 0.9004263014711802 | 0.9025422756131569 |
| roc_auc | 0.9658234827802413 | 0.9645021561146785 | 0.9639141705648404 |
| confusion_matrix | [[24643, 1083], [4183, 24782]] | [[5266, 247], [920, 5287]] | [[5271, 241], [901, 5305]] |
| n_samples | 54691 | 11720 | 11718 |

### et_top50

**VERIFIED** - Saved metrics:

| Metric | TRAIN | VAL | TEST |
| --- | --- | --- | --- |
| accuracy | 0.9133495456290798 | 0.9083617747440274 | 0.911247653183137 |
| balanced_accuracy | 0.9156776014458978 | 0.9107366974280602 | 0.9133393721699166 |
| malware_precision | 0.9563710345866928 | 0.9522466960352423 | 0.9506280530355897 |
| malware_recall | 0.8763680303814949 | 0.8706299339455453 | 0.8780212697389623 |
| malware_f1 | 0.9146233808348496 | 0.9096111765696011 | 0.9128832300217792 |
| macro_f1 | 0.9133302519023436 | 0.9083442627959617 | 0.9112163584445303 |
| roc_auc | 0.9731562844647906 | 0.9710228538132301 | 0.97042641736285 |
| confusion_matrix | [[24568, 1158], [3581, 25384]] | [[5242, 271], [803, 5404]] | [[5229, 283], [757, 5449]] |
| n_samples | 54691 | 11720 | 11718 |

## Runtime

**VERIFIED** - Seconds copied directly from per-set `runtime.json`. Final model time is the retained winning candidate's fit call on TRAIN alone, with no refit. Per-set search includes candidate fits, VAL prediction/scoring, warning capture and search logging; it excludes TEST. Root all-set search is outer-loop wall time, additionally including per-set preparation/metadata overhead, not a sum of final model fit times.

| Set | Final-model fit seconds | Feature-set search seconds |
| --- | --- | --- |
| mi_top50 | 0.1567271999992954 | 1.4346088000002055 |
| mi_top30 | 0.06837939999968512 | 0.6859557999996468 |
| mi_top100 | 0.5019833999995171 | 4.3204476999999315 |
| et_top50 | 0.13727470000048925 | 1.361035500000071 |

**VERIFIED** - Root `total_all_feature_sets_search_seconds`: `8.569824500000323`. These are measurements from this machine and environment, not portable speed guarantees.

## Every sklearn warning and its meaning

**VERIFIED** - There were eight captured sklearn warning occurrences in the successful search: one for each class at reg_param=0.0 in each feature set. All are `LinAlgWarning`, from `sklearn/discriminant_analysis.py`, line 1024 in the installed 1.7.2 environment. Each search JSON preserves the full local source path, category, message and line number. Exact messages, including repeated occurrences:

| Feature set | reg_param | Category | Exact message |
| --- | --- | --- | --- |
| mi_top50 | 0.0 | LinAlgWarning | The covariance matrix of class 0 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| mi_top50 | 0.0 | LinAlgWarning | The covariance matrix of class 1 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| mi_top30 | 0.0 | LinAlgWarning | The covariance matrix of class 0 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| mi_top30 | 0.0 | LinAlgWarning | The covariance matrix of class 1 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| mi_top100 | 0.0 | LinAlgWarning | The covariance matrix of class 0 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| mi_top100 | 0.0 | LinAlgWarning | The covariance matrix of class 1 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| et_top50 | 0.0 | LinAlgWarning | The covariance matrix of class 0 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |
| et_top50 | 0.0 | LinAlgWarning | The covariance matrix of class 1 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity. |

**VERIFIED** - All other search warning lists are empty. All four `evaluation_warnings.json` files have empty TRAIN, VAL and TEST lists. The selected candidates are regularised and emitted no captured fit or evaluation warnings.

**VERIFIED (library diagnostic)** - Rank deficiency means a class covariance matrix is not considered full rank under sklearn's tolerance. A feature can vary across TRAIN yet be constant within one class, and feature dependencies can also reduce covariance rank. Dropping only globally constant features does not guarantee full rank. Regularisation stabilises QDA's covariance estimate; it does not switch to another classifier.

**NOT VERIFIED** - We did not isolate which exact feature dependencies caused each class's warning, test multivariate Gaussian goodness of fit, or prove the unregularised candidates numerically reliable. A finite validation score is not proof of stable covariance inversion. The selected regularised candidates have no captured warnings; this is not a proof that their modelling assumptions hold.

**VERIFIED** - Separate setup/verification messages: pip initially encountered sandbox network restrictions and cache-deserialisation warnings; approved downloads and installation succeeded. A setup import of Matplotlib could not write its default user cache and used a temporary cache; the final run set `MPLCONFIGDIR` to the writable `.venv-sklearn172/matplotlib-cache`. Post-run CSV consistency verification emitted four Pandas4Warning messages that `float_precision` is deprecated. Those came from the verification command, not model fitting/evaluation, and did not alter outputs. No warnings were reported by the Step 6 pytest run.

## Comparison with GROUP 3 LightGBM

**VERIFIED** - Group 3 numbers below are copied from their `kd09_eval_metrics.json`; they are NOT Group 1 QDA results. Their `pipeline/kronodroid/stage09_eval.py` uses binary precision/recall/F1 with sklearn's default positive label 1 and rounds all four metrics to four decimal places before writing JSON. We preserve that stored precision and do not invent underlying digits. Our values remain full JSON values.

| Owner / model | Feature representation | TEST accuracy | TEST Malware F1 |
| --- | --- | --- | --- |
| Group 1 QDA primary mi_top50 | 50 MI features, Group 3 standardized kd05 | 0.8959720088752346 | 0.8959631304941538 |
| Group 1 QDA VAL-selected et_top50 | 50 ET features, Group 3 standardized kd05 | 0.911247653183137 | 0.9128832300217792 |
| GROUP 3 LightGBM replication | Same et_top50 identities, additional TRAIN-fitted MinMax | 0.9779 | 0.979 |
| GROUP 3 LightGBM full_standardized | All 475 standardized features | 0.9834 | 0.9842 |

**VERIFIED** - QDA scored lower on this held-out split. Both tracks consume the shared KronoDroid split; `stage08_train.py` confirms Group 3 replication uses et_top50, adds MinMax scaling and trains LightGBM with balanced class weights and VAL early stopping. Their full model uses all 475 standardized features. Our et_top50 uses the same feature identities but not the extra MinMax representation, class weights, estimator or early-stopping procedure. This is a measured comparison, not a controlled isolation of the effect of the classifier alone. No QDA settings were changed after inspecting TEST results.

**NOT VERIFIED** - We did not independently rerun Group 3 LightGBM or validate the published Random Forest benchmark. The cited benchmark entry is external attribution and is not a result of this QDA work. We do not claim statistical significance or that one model will always outperform another.

## Explicit assumptions and limitations

- **VERIFIED (declared assumption)** - Treat Group 3's processed artifacts and label mapping as the input authority; hashes and cross-artifact agreement were checked. **NOT VERIFIED** - Upstream labels were not independently audited against app behavior, and Group 3's raw preprocessing was not rerun.
- **VERIFIED (declared assumption)** - Exact floating-point equality defines a score tie; larger reg_param wins within a feature set. Exact cross-set ties follow the prespecified configured order. No tie epsilon or TEST metric participates in selection.
- **VERIFIED (declared convention)** - Undefined precision/recall/F1 uses zero_division=0. Both classes occur in every split. Positive label is Malware=1; default empirical priors are used rather than forced balanced priors.
- **VERIFIED** - Binary detection only; family_label is not used in X or y. No family model was attempted. Emulator data is not used. MalDroid is a different five-class dataset, and its results are not directly comparable.
- **VERIFIED** - One fixed supplied split and one successful QDA run were used. No confidence intervals, repeated-split validation, probability calibration study or external deployment evaluation was performed. **NOT VERIFIED** - Generalisation to future apps, temporal drift, emulator apps or other datasets.
- **NOT VERIFIED** - QDA's class-conditional multivariate Gaussian assumption was not established. Permission flags and transformed counts do not become Gaussian simply because they are standardized.
- **VERIFIED** - The existing MalDroid README claims guards for both Group 3 output directories, while its train_qda.py guards only runs/real. That pre-existing inconsistency was disclosed and left unchanged at the user's direction. The new KronoDroid guard protects both paths and its rejection behavior is tested.
- **VERIFIED** - The new output guard protects its write destinations; it does not globally make Group 3 files immutable to unrelated processes. This run reads those inputs and verifies their hashes before/after use. The one-evaluation rule is enforced for each winner within a run; the CLI does not prevent someone manually launching a completely new run later. Do not rerun TEST to tune this experiment.

## Artifacts and reproduction

**VERIFIED** - Root artifacts: `config_used.yaml`, `package_versions.json`, `random_seed`, `input_provenance.json`, `selection_summary.json`, `runtime.json`, `run_status.json`. Each feature-set subdirectory contains:

- `model.joblib`
- `config_used.yaml`
- `package_versions.json`
- `random_seed`
- `reg_param_search.json`
- `train_metrics.json`
- `val_metrics.json`
- `test_metrics.json`
- `test_predictions.csv`
- `confusion_matrix.png`
- `runtime.json`
- `evaluation_warnings.json`

**VERIFIED** - Model payload keys: `model`, `features` (ordered retained names), `dropped_features`, `classes`. Classes are [0,1]. SHA256 IDs in test_predictions.csv come from the unnamed kd05 index, exported as `sha256`; remaining columns are `true_label`, `predicted_label`, `prob_malware`. Each CSV contains 11,718 unique test IDs matching the manifest. Recomputing TEST metrics from the saved CSV exactly matched all numeric JSON metrics and the confusion matrix; that check did not call the models again.

**VERIFIED** - Read-only validation and Step 6 test commands used the 1.7.2 environment:

```powershell
.\.venv-sklearn172\Scripts\python.exe groups/group1_qda/kronodroid/src/run_qda_kronodroid.py --config groups/group1_qda/kronodroid/configs/qda_kronodroid.yaml --validate-only
.\.venv-sklearn172\Scripts\python.exe -m pytest groups/group1_qda/kronodroid/tests/test_kronodroid_qda.py -v
```

**VERIFIED** - Actual Step 6 output:

```text
platform win32 -- Python 3.14.3, pytest-9.2.0
collecting ... collected 26 items
============================= 26 passed in 3.75s ==============================
```

**VERIFIED** - Tests cover real split separation, deliberate duplicate-ID rejection, all six top-K subsets/order and metadata exclusion, missing/reordered feature-list rejection, both protected output paths including normalized traversal, refusal before mkdir, TRAIN-only constant filtering and selection tie rules. An autouse fixture makes any QDA fit/predict/predict_proba call fail the test, so pytest does not reevaluate TEST.

**VERIFIED (historical command)** - The successful training/evaluation command was run once in the new environment. This command performs TEST evaluation; it is recorded for provenance, not an instruction to rerun it during report review:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.venv-sklearn172/matplotlib-cache'
.\.venv-sklearn172\Scripts\python.exe groups/group1_qda/kronodroid/src/run_qda_kronodroid.py --config groups/group1_qda/kronodroid/configs/qda_kronodroid.yaml
```

**VERIFIED** - To recreate a separate environment later, install exact `name==version` entries from this run's `package_versions.json` using Python 3.14.3; keep that JSON with the report. No requirement to copy or commit a virtual environment. The present results already exist and can be reviewed from JSON/CSV without training.

## Critical review and delivery status at completion of Step 7

**VERIFIED** - Critical review checks: all 12 split-metric JSONs and all 32 search records are transcribed without rounding; selection maxima, dropped-feature lists, counts, runtime fields, package versions and eight warning occurrences are checked against saved artifacts. Input hashes are rechecked. Saved model feature lists, class means and empirical priors were independently checked against TRAIN data without fitting or prediction. The report distinguishes primary from VAL-selected model, Malware F1 from macro F1, fit time from search time, and Group 3's rounded numbers from our full-precision JSON values. Version-change history and both the code-level and modelling limitations are explicit. Unsupported claims about Gaussian fit, future performance and comparative significance are marked NOT VERIFIED.

**VERIFIED** - The existing tracked MalDroid files and Group 3 artifacts show no tracked modifications. No commit, model staging or push has been performed. `.gitignore` ignores prediction CSVs and joblib models, so stage prediction CSVs explicitly with git add -f during the approved delivery step. Ask the user before committing model.joblib. Step 8 must first show git status and proposed files, then obtain approval before commit/push to origin/grp1-qda-kronodroid; never push master or DeepMehta06.
