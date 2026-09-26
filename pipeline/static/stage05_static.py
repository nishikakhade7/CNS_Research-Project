"""Stage 05 — Static Branch Preprocessing.

Consumes Stage 03's typing decision (numeric-usable columns only — text/high-cardinality/
zero-variance columns were already identified and excluded, never silently coerced).
Applies log1p to count-like columns and fits a StandardScaler — both fit on the TRAIN split
only, then applied unchanged to val/test. This is the train-only-fitting rule in code, not
just in the documentation.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage05_static")


class StaticPreprocessingStage(Stage):
    name = "stage05_static"
    depends_on = ["stage03_typing", "stage04_split"]

    def config_section(self, cfg: dict) -> dict:
        return {"paths": {"maldroid_static_repaired": cfg["paths"]["maldroid_static_repaired"]}}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        static_path = resolve(cfg["paths"]["maldroid_static_repaired"])

        with open(ctx.run_dir / "artifacts" / "stage03_static_feature_types.json", encoding="utf-8") as f:
            typing_report = json.load(f)
        col_types = typing_report["column_types"]
        idx_col = next(c for c, t in col_types.items() if t == "identity_row_index")
        count_cols = [c for c, t in col_types.items() if t == "count_or_continuous_numeric"]
        binary_cols = [c for c, t in col_types.items() if t == "binary_numeric"]
        usable_cols = count_cols + binary_cols
        log.info(f"Static branch: {len(usable_cols)} usable numeric columns "
                 f"({len(count_cols)} count-like, {len(binary_cols)} binary).")

        split_manifest = pd.read_parquet(ctx.run_dir / "manifests" / "maldroid_split_manifest.parquet")

        pf = pq.ParquetFile(str(static_path))
        table = pf.read(columns=[idx_col] + usable_cols)
        df = table.to_pandas()
        df = df.rename(columns={idx_col: "sample_id"})
        df[usable_cols] = df[usable_cols].fillna(0)

        df = df.merge(split_manifest[["sample_id", "split"]], on="sample_id", how="inner")
        train_mask = df["split"] == "train"

        log.info(f"Applying log1p to {len(count_cols)} count-like columns (no fitting required)...")
        import numpy as np
        if count_cols:
            df[count_cols] = df[count_cols].clip(lower=0)
            df[count_cols] = np.log1p(df[count_cols])

        log.info(f"Fitting StandardScaler on TRAIN split only ({train_mask.sum()} rows)...")
        scaler = StandardScaler()
        scaler.fit(df.loc[train_mask, usable_cols])
        scaled = scaler.transform(df[usable_cols])
        scaled_df = pd.DataFrame(scaled, columns=[f"{c}__scaled" for c in usable_cols], index=df.index)

        out_df = pd.concat([df[["sample_id", "split"]], df[usable_cols], scaled_df], axis=1)
        out_path = ctx.run_dir / "artifacts" / "stage05_static_features.parquet"
        out_df.to_parquet(out_path, index=False)

        scaler_path = ctx.run_dir / "artifacts" / "stage05_static_scaler.joblib"
        joblib.dump({"scaler": scaler, "raw_columns": usable_cols, "count_columns": count_cols}, scaler_path)

        log.info(f"Static branch preprocessed matrix: {out_df.shape} -> {out_path}")
        return {
            "static_features": str(out_path),
            "scaler_artifact": str(scaler_path),
            "usable_columns": len(usable_cols),
            "rows": int(len(out_df)),
        }
