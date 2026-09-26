# Group 3 Pipeline — Usage

## Environment

This pipeline runs in the `ml_cuda` conda environment (Python 3.10, PyTorch 2.5.1+CUDA,
verified RTX 4060 GPU). Activate it before running anything:

```
conda activate ml_cuda
```

Install any missing packages with `pip install -r requirements.txt` inside that environment.

## Running

```
python one_run.py --all                          # run every stage in order
python one_run.py --stage stage01_audit           # run a single stage
python one_run.py --stage stage03_typing --force-stage   # force re-run even if already succeeded
python one_run.py --list                          # list all registered stage names
```

Every stage is resumable: if it already succeeded with the same configuration, it is
skipped automatically. If a stage's prerequisite has not succeeded, the run hard-stops with
a clear error naming the missing stage — nothing is silently re-derived or skipped.

## Stage order

| Stage | Name | Depends on |
|---|---|---|
| 01 | `stage01_audit` | — |
| 02 | `stage02_manifest` | 01 |
| 03 | `stage03_typing` | 02 |
| 04 | `stage04_split` | 02 |
| 05 | `stage05_static` | 03, 04 |
| 06 | `stage06_dynamic` | 04 |
| 07 | `stage07_fusion` | 05, 06 |
| 08 | `stage08_selection` | 05, 06, 07 |
| 09 | `stage09_imbalance` | 08 |
| 11 | `stage11_train` | 08, 09 |
| 12 | `stage12_eval` | 11 |
| 13 | `stage13_ablation` | 08, 09 |
| 14 | `stage14_lock` | 13 |
| 15 | `stage15_external` | 14 |
| 16 | `stage16_report` | 12, 13, 15 |

(Stage 10 — model-specific adapters — is implemented as reusable modules under
`pipeline/adapters/`, consumed by Stages 11/13/14 rather than being its own CLI stage.)

## Output layout

```
runs/latest/
    status.json              per-stage status, timing, config hash, artifact paths
    config.yaml.snapshot.json
    environment.json         captured hardware/package/CUDA state for this run
    logs/run.log, error.log
    manifests/               canonical + split manifests (parquet)
    artifacts/               preprocessed feature matrices, scalers, selection reports, frozen model
    checkpoints/             trained model files (.joblib, .pt)
    metrics/                 training/evaluation/ablation JSON results
    plots/                   PNG plots
    reports/FINAL_REPORT.md  consolidated final report
```

## Config

`config/default.yaml` — all paths, split fractions, label maps, feature-selection methods/K
values, imbalance strategy, and per-model hyperparameters live here. Pass `--config path.yaml`
to use an alternate configuration.
