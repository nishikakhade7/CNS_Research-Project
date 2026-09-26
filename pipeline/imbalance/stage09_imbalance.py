"""Stage 09 — Class Imbalance Handling (TRAIN ONLY).

Computes, per modality, everything a training stage needs to apply an imbalance strategy
without ever touching validation/test distributions:
  - original class distribution (train/val/test, reported, never altered)
  - sklearn 'balanced' class weights (computed from TRAIN labels only)
  - undersampled TRAIN sample_ids (majority classes downsampled to the smallest class size)
  - SMOTE-resampled TRAIN arrays, where dimensionality makes SMOTE tractable

SMOTE is skipped for the raw static branch (50,621-ish raw columns before any selection) —
applying k-NN-based interpolation in a space that wide is not a trustworthy operation and
is logged as a documented limitation, not silently attempted or faked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.selection.stage08_selection import MODALITY_ARTIFACTS, _get_labels, _raw_feature_columns

log = get_stage_logger("stage09_imbalance")

SMOTE_MAX_RAW_DIM = 600  # static (~50k raw cols) exceeds this; dynamic/fused variants do not


class ImbalanceStage(Stage):
    name = "stage09_imbalance"
    depends_on = ["stage08_selection"]

    def config_section(self, cfg: dict) -> dict:
        return {"imbalance": cfg["imbalance"], "modalities": cfg["modalities"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        results: dict[str, Any] = {}

        for modality in cfg["modalities"]:
            path = ctx.run_dir / "artifacts" / MODALITY_ARTIFACTS[modality]
            df = pd.read_parquet(path)
            y_all = _get_labels(df, ctx.run_dir)
            feat_cols = _raw_feature_columns(df, modality)

            dist = {}
            for split_name in ("train", "val", "test"):
                mask = (df["split"] == split_name).to_numpy()
                dist[split_name] = y_all[mask].value_counts().to_dict()

            train_mask = (df["split"] == "train").to_numpy()
            y_train = y_all[train_mask].to_numpy()
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_dict = {str(c): float(w) for c, w in zip(classes, weights)}

            min_count = int(pd.Series(y_train).value_counts().min())
            undersample_ids = []
            for c in classes:
                ids_c = df.loc[train_mask & (y_all == c).to_numpy(), "sample_id"].to_numpy()
                rng = np.random.RandomState(cfg["seed"])
                chosen = rng.choice(ids_c, size=min_count, replace=False)
                undersample_ids.extend(chosen.tolist())

            modality_result: dict[str, Any] = {
                "distributions": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in dist.items()},
                "class_weights": class_weight_dict,
                "undersample_train_sample_ids": [int(x) for x in undersample_ids],
            }

            if len(feat_cols) <= SMOTE_MAX_RAW_DIM:
                X_train = df.loc[train_mask, feat_cols].to_numpy(dtype=float)
                smote = SMOTE(random_state=cfg["seed"])
                X_res, y_res = smote.fit_resample(X_train, y_train)
                smote_path = ctx.run_dir / "artifacts" / f"stage09_smote_{modality}.npz"
                np.savez_compressed(smote_path, X=X_res, y=y_res, columns=np.array(feat_cols))
                modality_result["smote_artifact"] = str(smote_path)
                modality_result["smote_resampled_size"] = int(len(y_res))
                log.info(f"[{modality}] SMOTE applied: {len(y_train)} -> {len(y_res)} train rows")
            else:
                modality_result["smote_artifact"] = None
                modality_result["smote_skipped_reason"] = (
                    f"{len(feat_cols)} raw features exceeds SMOTE_MAX_RAW_DIM={SMOTE_MAX_RAW_DIM}; "
                    "k-NN interpolation in this wide a raw space is not trustworthy before feature "
                    "selection. Use SMOTE on a feature-selected subset at training time instead."
                )
                log.warning(f"[{modality}] SMOTE skipped: {modality_result['smote_skipped_reason']}")

            results[modality] = modality_result
            log.info(f"[{modality}] train distribution: {dist['train']}, class_weights: {class_weight_dict}")

        import json
        out_path = ctx.run_dir / "artifacts" / "stage09_imbalance_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        return {"imbalance_report": str(out_path), "modalities": list(results.keys())}
