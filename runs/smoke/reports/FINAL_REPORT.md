# Final Reproducibility & Results Report

**Locked configuration:** model=`lightgbm`, modality=`dynamic_139`, features=8, class_weight_applied=True, internal macro-F1 at selection=0.1600
**Independently re-verified on TEST after retrain:** macro-F1=0.1600, accuracy=0.2222 — MATCHES Stage 13

## Internal Evaluation (MalDroid TEST split, dynamic_470)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---|---|---|---|
| lightgbm | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Modality Ablation (LightGBM, TEST split)

| Modality | Macro-F1 | Accuracy | Feature Count | Class Weight Applied |
|---|---|---|---|---|
| static | 0.0000 | 0.0000 | 8 | True |
| dynamic_139 | 0.1600 | 0.2222 | 8 | True |
| dynamic_470 | 0.0000 | 0.0000 | 8 | True |
| fused | 0.0000 | 0.0000 | 8 | True |

## Known Data-Quality Disclosure

No cross-class duplicate-vector conflicts found in this run.

## External (AndMal2020) Validation

**Not possible for the locked configuration.** Reason: The frozen model requires its exact trained feature columns to be present in AndMal2020 by name. MalDroid's dynamic features are syscall/binder names (e.g. process/IPC-level events); AndMal2020's dynamic features are memory/API/battery/logcat/process-count names — different instrumentation entirely. Zero-filling the missing columns was considered and rejected: it would silently change what the model is being asked to classify, not perform a real evaluation. This is reported as a genuine cross-dataset feature-space incompatibility, not fabricated as a passing external-validation score.

## Plots

- `plots\confusion_matrix_lightgbm.png`
- `plots\internal_macro_f1_by_model.png`
- `plots\modality_comparison_lightgbm.png`
- `plots\training_time_by_model.png`