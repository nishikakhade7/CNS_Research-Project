# Final Reproducibility & Results Report

**Locked configuration:** model=`lightgbm`, modality=`fused`, features=270, class_weight_applied=True, internal macro-F1 at selection=0.9795
**Independently re-verified on TEST after retrain:** macro-F1=0.9795, accuracy=0.9819 — MATCHES Stage 13

## Internal Evaluation (MalDroid TEST split, dynamic_470)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---|---|---|---|
| lightgbm | 0.9543 | 0.9457 | 0.9543 | 0.9478 |

## Modality Ablation (LightGBM, TEST split)

| Modality | Macro-F1 | Accuracy | Feature Count | Class Weight Applied |
|---|---|---|---|---|
| static | 0.9750 | 0.9776 | 270 | True |
| dynamic_139 | 0.9308 | 0.9440 | 139 | True |
| dynamic_470 | 0.9457 | 0.9543 | 270 | True |
| fused | 0.9795 | 0.9819 | 270 | True |

## Known Data-Quality Disclosure

**3 group(s)** of samples share an identical 470-feature dynamic vector but carry *different* ground-truth labels (e.g. same behavioral fingerprint labeled both Adware and Riskware). Each group is kept entirely within one split (no train/test leakage), but this represents a genuine label-ambiguity ceiling in this feature representation — disclose alongside any accuracy/macro-F1 figure derived from it.
- dup_group=5: classes=['Adware', 'Riskware'], resolved to split=`train`
- dup_group=776: classes=['Adware', 'Riskware'], resolved to split=`test`
- dup_group=1257: classes=['Banking', 'SMS Malware', 'Riskware', 'Benign'], resolved to split=`train`

## External (AndMal2020) Validation

**Not possible for the locked configuration.** Reason: The frozen model requires its exact trained feature columns to be present in AndMal2020 by name. MalDroid's dynamic features are syscall/binder names (e.g. process/IPC-level events); AndMal2020's dynamic features are memory/API/battery/logcat/process-count names — different instrumentation entirely. Zero-filling the missing columns was considered and rejected: it would silently change what the model is being asked to classify, not perform a real evaluation. This is reported as a genuine cross-dataset feature-space incompatibility, not fabricated as a passing external-validation score.

## Plots

- `plots\confusion_matrix_lightgbm.png`
- `plots\internal_macro_f1_by_model.png`
- `plots\modality_comparison_lightgbm.png`
- `plots\training_time_by_model.png`