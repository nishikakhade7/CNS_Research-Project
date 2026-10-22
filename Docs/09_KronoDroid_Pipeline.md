# 09 — KronoDroid Pipeline (Group 3, second dataset track)

Every number in this document is either (a) read directly from `runs/kronodroid/status.json` /
`runs/kronodroid/metrics/kd09_eval_metrics.json` after an actual run of `kronodroid_run.py --all`,
or (b) attributed to a named paper. Nothing here is projected or estimated. `CONFIRMED` unless
marked `CITED` or `PROPOSED`.

This is a **separate, parallel pipeline** to the existing MalDroid2020/AndMal2020 pipeline
(`one_run.py`). It does not touch, re-run, or modify that pipeline or its results — those remain
closed and final (98% F1, documented in `Docs/final/`). Entry point: `kronodroid_run.py`. Config:
`config/kronodroid.yaml`. Stage code: `pipeline/kronodroid/`. Run artifacts: `runs/kronodroid/`.

## 0. What this builds on

Full forensic detail for everything in this section lives in
[`Docs/02_Deep_Dataset_Forensic_Findings.md`](02_Deep_Dataset_Forensic_Findings.md), Part C — read
that first. Summary of the decisions this pipeline encodes:

- **real_device only.** Emulator shares 80% of its apps with real_device (63,320 of 78,800 unique
  `sha256` across both) — not an independent dataset. Documented, not used.
- **4 label-conflict hashes dropped.** Same `sha256` filed as both malware and legitimate,
  consistently across real_device and emulator. Upstream labeling defect, not resolvable by
  picking a side.
- **wc -l / line-counting tools never used on these files.** Every row's `Package` field has an
  embedded newline, which doubles naive line counts. Always `pandas.read_csv` or equivalent.

## 1. Sources cited

1. Guerra-Manzanares, Bahsi, Nõmm. "KronoDroid: Time-based Hybrid-featured Dataset for Effective
   Android Malware Detection and Characterization." *Computers & Security*, Vol 110, 102399, 2021.
   — original dataset paper. Cited for provenance, the static+dynamic feature design intent, and
   the real_device/emulator distinction. **Not** cited for row counts without independent
   re-verification (done in Doc 02 Part C).
2. Waheed & Qadir. "Effective and Efficient Android Malware Detection and Category Classification
   Using the Enhanced KronoDroid Dataset." *Security and Communication Networks*, 2024. — cited
   methodology and benchmark numbers for the real_device split, used as a comparison target in
   Section 7, never copied as if they were our own result.

## 2. Stage 01 — Audit (`kd01_audit`)

Re-measures the Doc 02 claims against the live files (hard-stops if the raw files ever drift from
what was forensically audited) and drops the 4 label-conflict hashes.

| Metric | Value |
|---|---|
| real_malware_v1.csv rows (re-confirmed) | 41,382 |
| real_legitimate_v1.csv rows (re-confirmed) | 36,755 |
| Raw combined rows | 78,137 |
| Rows dropped (4 label-conflict hashes × 2 files) | 8 |
| Clean combined rows | **78,129** |

## 3. Stage 02 — Manifest (`kd02_manifest`)

One row per sample: `sample_id` (= `sha256`), binary label, and a thresholded family label.

| Metric | Value |
|---|---|
| Total samples | 78,129 |
| Malware | 41,378 |
| Benign | 36,751 |
| Raw `MalFamily` values (malware only) | 240 |
| Family-min-sample threshold | 100 |
| Family classes kept at that threshold | 35 (+ "Other" + "Benign") |

Top family buckets after thresholding: Benign 36,751; Airpush/StopSMS 7,775; SMSreg 5,019; Malap
4,055; Boxer 3,597; **Other 3,033** (everything below the 100-sample threshold, or missing
`MalFamily`); Agent 2,934; FakeInst 2,384; Locker/SLocker Ransomware 1,846; BankBot 1,297.

**This is NOT a replication of Waheed & Qadir's 15-category VirusTotal/engine-consensus taxonomy.**
That mapping requires their specific engine-consensus labeling (VirusTotal/F-Secure/FortiGuard
cross-referencing), which is not available to us as a reusable artifact — inventing a plausible-
looking version of it would be fabrication. Instead this is this project's own documented
contribution (see Section 6): a data-driven minimum-sample threshold over the raw long tail,
exactly where the cited paper flagged future work. **Multi-class training on this family label is
not yet run** — the threshold and bucket counts above are measured; a trained family-level model is
future work, stated explicitly rather than invented (see Section 9).

## 4. Stage 03 — Column manifest (`kd03_typing`)

Classified all 484 columns by name-matching against reference lists (ARM EABI syscall names,
`android.Manifest.permission` constants — see `pipeline/kronodroid/reference_data.py` for the exact
lists and their declared sources), not by column position.

| Group | Count |
|---|---|
| Dynamic — known syscall name | 217 |
| Dynamic — reserved/summary syscall slots (`SYS_300`–`SYS_369`, `syscall_983042`, `nr_syscalls`) | 72 |
| Dynamic — behavioral (`NrContactedIps`) | 1 |
| **Total dynamic** | **290** |
| Static — Android permission flag | 166 |
| Static — permission summary (`nr_permissions`, `dangerous`, `total_perm`, …) | 7 |
| Static — APK structural (file size, component/intent counts) | 12 |
| **Total static** | **185** |
| Metadata / identifiers (never fed to the model) | 9 |
| **Unclassified** | **0** |
| **Total** | **484** |

Comparison to the paper's claimed 200 static + 289 dynamic: our measured dynamic count (290)
matches almost exactly — the 1-column gap is `nr_syscalls`, itself a summary column over the
syscall block, not a named syscall; excluding it gives exactly 289. Static is measured at 185,
15 short of the paper's claimed 200 — the gap is not resolved (the paper does not publish its exact
column list); documented as a measured discrepancy, not forced to match.

## 5. Stage 04 — Leakage-safe split (`kd04_split`)

70/15/15 train/val/test, stratified on the binary label, grouped by `sha256` (each sample is
already its own group post-audit — Stage 01 guarantees zero duplicate `sha256`). Post-split
assertion confirms no `sha256` spans more than one split.

**Deviation from the cited paper's 70/30 (no val, 5-fold CV on the 70% instead):** kept this
project's existing three-way discipline (MalDroid and AndMal both use train/val/test) so the val
split is a genuine held-out set used for early stopping, distinct from the test split used only
once for final reporting — not a CV fold reused across model selection and evaluation. `PROPOSED`

| Split | Benign | Malware | Total |
|---|---|---|---|
| train | 25,726 | 28,965 | 54,691 |
| val | 5,513 | 6,207 | 11,720 |
| test | 5,512 | 6,206 | 11,718 |

## 6. Stage 05–07 — Preprocessing, selection, imbalance

- **Stage 05 (preprocess):** `log1p` on the 290 count-like dynamic columns, then `StandardScaler`
  fit on TRAIN ONLY, applied to val/test (same train-only-fit discipline as the MalDroid pipeline's
  stage05/06). 475 feature columns total (290 dynamic + 185 static).
- **Stage 06 (selection):** replicates Waheed & Qadir's filter-method baseline — `ExtraTrees`
  importance + mutual information, MinMax-scaled (their best-reported config), ranked over all 475
  features, top-30/50/100 compared. Measured overlap between the two methods' top-50 sets: **31 of
  50** features in common — the two filter methods agree on about 62% of their top picks, not
  identical rankings.
- **Stage 07 (imbalance, binary):** measured TRAIN-split ratio is **1.126×** (28,965 malware vs.
  25,726 benign) — confirms Doc 02's near-balanced finding. `class_weight="balanced"` applied at
  training time; no SMOTE. **Family-level imbalance is this project's own contribution area** (the
  cited paper explicitly left it as future work): the Stage 02 "Other" bucket + the eventual
  family-level `class_weight` is the mechanism, but — see Section 3 — the family model itself is
  not yet trained. `PROPOSED` for the untrained part, `CONFIRMED` for the binary ratio.

## 7. Stage 08–09 — Training and evaluation (binary, real measured results)

Two LightGBM models trained on TRAIN, early-stopped on VAL, scored once on the untouched TEST split
(11,718 samples: 6,206 malware, 5,512 benign):

| Model | Features | Scaling | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `replication` (closest reproduction of Waheed & Qadir's best config) | Top-50, ExtraTrees-selected | MinMax | 0.9779 | 0.9836 | 0.9745 | 0.9790 |
| `full_standardized` (this project's own default) | All 475 | StandardScaler | **0.9834** | **0.9875** | **0.9810** | **0.9842** |
| **Cited: Waheed & Qadir (2024), Random Forest** `CITED` | Top-50, ExtraTrees-selected | MinMax | 0.9803 | 0.9851 | 0.9773 | 0.9812 |

**This is not an apples-to-apples comparison**, for the same reason Doc 02/Docs/final's external
validation doc insists on stating the caveat plainly: different model (LightGBM vs. their
GridSearchCV-tuned Random Forest), different split protocol (our fixed 70/15/15 vs. their 70/30 +
5-fold CV), and our own early-stopping/val-based model selection. What the comparison does show,
measured: our `full_standardized` LightGBM, with no feature-selection restriction at all, scores
higher on every metric than both the cited RF benchmark and our own closest reproduction of their
exact feature-selection recipe — consistent with LightGBM's generally stronger performance on
tabular data with large feature counts, not proof that our pipeline is "better" in any causally
isolated sense.

## 8. Reproducing this run

```powershell
conda activate ml_cuda   # matches runs/real/environment.json: lightgbm 4.7.0, sklearn 1.7.2, pandas 2.3.3
python kronodroid_run.py --all
python kronodroid_run.py --list   # see all registered stage names
```

Raw files expected at `Dataset/KronoDroid/real_device/*.csv` (see
[`Docs/DATASET_SETUP.md`](DATASET_SETUP.md)). Each stage is resumable via config-hash-based
skip-if-unchanged, same mechanism as `one_run.py` (`pipeline/core/stage.py` is shared, unmodified,
between both pipelines).

## 9. Explicitly not done in this pass (future work, not fabricated)

- **Family-level multi-class model is not trained.** The 35-class-plus-Other label exists in the
  Stage 02 manifest (measured), but no LightGBM model has been fit or evaluated on it. Any
  macro-F1 number for this track would be invented — not reported until actually run.
- **No faithful reproduction of Waheed & Qadir's 15-category taxonomy.** Requires their VirusTotal/
  engine-consensus mapping table, which is not available to us; see Section 3.
- **Emulator track is unused**, by the Doc 02 Part C.6 decision — not a gap, a documented choice.
- **No SHAP/feature-importance deep-dive, no hyperparameter search beyond the fixed config in
  `config/kronodroid.yaml`** — the LightGBM hyperparameters here are the same fixed defaults as
  this project's MalDroid pipeline, not separately tuned for KronoDroid.
