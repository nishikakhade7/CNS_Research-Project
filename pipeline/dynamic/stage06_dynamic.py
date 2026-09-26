"""Stage 06 — Dynamic Branch Preprocessing (both 139-feature and 470-feature views).

Both representations are processed as separate, parallel arms — the 470-feature view is
NOT assumed superior; Stage 13's ablation is what actually decides that empirically.
All columns here are already known (Doc 02 forensic audit) to be clean numeric frequency
counts with no missing values and no zero-variance columns, so no typing pass is needed —
only a log1p transform (heavy-tailed counts) and a train-only-fit StandardScaler.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from sklearn.preprocessing import StandardScaler

log = get_stage_logger("stage06_dynamic")


def _process_one(path, split_manifest: pd.DataFrame, run_dir, tag: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != "Class"]
    df.insert(0, "sample_id", range(len(df)))
    df = df.merge(split_manifest[["sample_id", "split"]], on="sample_id", how="inner")
    train_mask = df["split"] == "train"

    df[feat_cols] = np.log1p(df[feat_cols].clip(lower=0))

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feat_cols])
    scaled = scaler.transform(df[feat_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}__scaled" for c in feat_cols], index=df.index)

    out_df = pd.concat([df[["sample_id", "split", "Class"]], df[feat_cols], scaled_df], axis=1)
    out_path = run_dir / "artifacts" / f"stage06_dynamic_{tag}_features.parquet"
    out_df.to_parquet(out_path, index=False)

    scaler_path = run_dir / "artifacts" / f"stage06_dynamic_{tag}_scaler.joblib"
    joblib.dump({"scaler": scaler, "raw_columns": feat_cols}, scaler_path)

    log.info(f"Dynamic-{tag} branch preprocessed: {out_df.shape} -> {out_path}")
    return {"features": str(out_path), "scaler": str(scaler_path), "feature_count": len(feat_cols), "rows": int(len(out_df))}


class DynamicPreprocessingStage(Stage):
    name = "stage06_dynamic"
    depends_on = ["stage04_split"]

    def config_section(self, cfg: dict) -> dict:
        return {"paths": {
            "maldroid_dynamic_139": cfg["paths"]["maldroid_dynamic_139"],
            "maldroid_dynamic_470": cfg["paths"]["maldroid_dynamic_470"],
        }}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        split_manifest = pd.read_parquet(ctx.run_dir / "manifests" / "maldroid_split_manifest.parquet")

        log.info("Processing dynamic-139 (syscall-only) branch...")
        r139 = _process_one(resolve(cfg["paths"]["maldroid_dynamic_139"]), split_manifest, ctx.run_dir, "139")

        log.info("Processing dynamic-470 (syscall+binder, primary) branch...")
        r470 = _process_one(resolve(cfg["paths"]["maldroid_dynamic_470"]), split_manifest, ctx.run_dir, "470")

        return {"dynamic_139": r139, "dynamic_470": r470}
