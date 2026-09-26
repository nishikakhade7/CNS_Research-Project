"""Stage 11 — Model Training.

GROUP 3 SCOPE: this stage trains and owns LightGBM only. QDA, NuSVC, Balanced Random
Forest, and the Autoencoder are separate groups' models — their adapter modules remain in
`pipeline/adapters/` (other groups may build their own stages against the same shared
preprocessing artifacts this pipeline produces), but Group 3's own training/evaluation/
ablation/lock stages do not fit or checkpoint them.

Trains on the dynamic-470 modality (the primary view — Doc 05) using Stage 08's train-only
feature ranking and Stage 09's train-only imbalance handling (class weights). Records
training time, inference time (on the val split), feature count, and device used.
"""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.adapters import lightgbm_adapter
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.selection.stage08_selection import _get_labels

log = get_stage_logger("stage11_train")

PRIMARY_MODALITY = "dynamic_470"
PRIMARY_ARTIFACT = "stage06_dynamic_470_features.parquet"
DEFAULT_TREE_K = 270


def _load_ranked_features(ctx: StageContext, modality: str) -> list[str]:
    """Returns Stage 08's lightgbm_importance ranking for this modality, using the largest
    configured K available rather than a hardcoded literal key — Stage 08 only stores a
    JSON key per K value actually listed in config.feature_selection.k_values, so looking
    up a hardcoded "270" would KeyError on any config that doesn't happen to configure
    exactly 270 (e.g. a smoke-test config with tiny feature counts). Bug found and fixed
    during smoke-test preparation, not during the earlier real-data run."""
    with open(ctx.run_dir / "artifacts" / "stage08_feature_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    methods = sel[modality]["methods"]["lightgbm_importance"]
    best_k_key = max(methods.keys(), key=lambda k: int(k))
    return methods[best_k_key]


class TrainingStage(Stage):
    name = "stage11_train"
    depends_on = ["stage08_selection", "stage09_imbalance"]

    def config_section(self, cfg: dict) -> dict:
        return {"models": {"lightgbm": cfg["models"]["lightgbm"]}, "imbalance": cfg["imbalance"], "seed": cfg["seed"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        env = capture_environment()
        gpu_verified = env["lightgbm_gpu"]["supported"]

        df = pd.read_parquet(ctx.run_dir / "artifacts" / PRIMARY_ARTIFACT)
        ranked = _load_ranked_features(ctx, PRIMARY_MODALITY)
        k = min(DEFAULT_TREE_K, len(ranked))

        # Use the canonical STRING label (e.g. "Adware"), not the raw numeric Class column,
        # so this stage's labels are consistent with Stages 08/09/13/14 — all of which use
        # `_get_labels`. An earlier version read `df["Class"]` (int codes) directly here,
        # which crashed against Stage 09's now-string-keyed class_weights dict. Caught via
        # smoke test.
        y_all = _get_labels(df, ctx.run_dir)
        train_mask = (df["split"] == "train").to_numpy()
        val_mask = (df["split"] == "val").to_numpy()
        y_train_full = y_all[train_mask].to_numpy()
        y_val = y_all[val_mask].to_numpy()

        with open(ctx.run_dir / "artifacts" / "stage09_imbalance_report.json", encoding="utf-8") as f:
            imb = json.load(f)[PRIMARY_MODALITY]
        class_weight = dict(imb["class_weights"].items())

        model_dir = ctx.run_dir / "checkpoints"

        raw_cols = lightgbm_adapter.select_view(ranked, k)
        X_train = df.loc[train_mask, raw_cols].to_numpy()
        X_val = df.loc[val_mask, raw_cols].to_numpy()
        model, device_used = lightgbm_adapter.build_model(cfg, gpu_verified)

        t0 = time.time()
        model.fit(X_train, y_train_full, sample_weight=np.array([class_weight[y] for y in y_train_full]))
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_val)
        infer_time = time.time() - t0

        joblib.dump(model, model_dir / "lightgbm_dynamic470.joblib")

        results = {
            "lightgbm": {
                "device_used": device_used, "feature_count": len(raw_cols),
                "train_time_sec": round(train_time, 3), "inference_time_sec": round(infer_time, 4),
                "val_accuracy": float((preds == y_val).mean()),
            }
        }
        log.info(f"LightGBM trained on {device_used}: val_accuracy={results['lightgbm']['val_accuracy']:.4f} "
                 f"({len(raw_cols)} features, {train_time:.2f}s train)")

        out_path = ctx.run_dir / "metrics" / "stage11_training_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {"training_results": str(out_path), "models_trained": ["lightgbm"], "modality": PRIMARY_MODALITY}
