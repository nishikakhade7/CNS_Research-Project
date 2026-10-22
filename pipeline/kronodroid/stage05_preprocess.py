"""KronoDroid Stage 05 — feature preprocessing: log1p on count-like dynamic (syscall)
columns, then StandardScaler fit on TRAIN ONLY and applied to val/test (same train-only-fit
discipline as pipeline/static/stage05_static.py and pipeline/dynamic/stage06_dynamic.py for
MalDroid — logic pattern reused, not the module itself, since KronoDroid has one unified
file per sample rather than separate static/dynamic files to fuse).
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.core.stage import Stage, StageContext


class KronoDroidPreprocessStage(Stage):
    name = "kd05_preprocess"
    depends_on = ["kd03_typing", "kd04_split"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        audited = pd.read_parquet(ctx.run_dir / "artifacts" / "kd01_audited_real_device.parquet")
        split_manifest = pd.read_parquet(ctx.run_dir / "manifests" / "kd04_split_manifest.parquet")
        with open(ctx.run_dir / "manifests" / "kd03_column_manifest.json", encoding="utf-8") as f:
            colman = json.load(f)

        dynamic_cols = (
            colman["dynamic_syscall_known_name"]
            + colman["dynamic_syscall_reserved_or_summary"]
            + colman["dynamic_behavioral"]
        )
        static_cols = (
            colman["static_permission"]
            + colman["static_permission_summary"]
            + colman["static_apk_structural"]
        )
        feature_cols = dynamic_cols + static_cols

        df = audited.set_index("sha256")
        split_df = split_manifest.set_index("sample_id")[["split", "label_binary_raw"]]
        joined = df[feature_cols].join(split_df, how="inner")
        if len(joined) != len(split_manifest):
            raise ValueError(
                f"join mismatch: {len(joined)} rows after joining features to split manifest, "
                f"expected {len(split_manifest)}"
            )

        X = joined[feature_cols].astype(float).fillna(0.0)

        # log1p only on syscall/behavioral count columns (always >= 0); permissions are
        # already 0/1 flags and APK-structural counts are heterogeneous scales handled by
        # the scaler alone, same reasoning as MalDroid's stage06 dynamic preprocessing.
        count_like_cols = [c for c in dynamic_cols if (X[c] >= 0).all()]
        X_logged = X.copy()
        X_logged[count_like_cols] = np.log1p(X_logged[count_like_cols])

        train_mask = joined["split"] == "train"
        scaler = StandardScaler()
        scaler.fit(X_logged.loc[train_mask, feature_cols])
        X_scaled = pd.DataFrame(
            scaler.transform(X_logged[feature_cols]), index=X_logged.index, columns=feature_cols
        )

        artifacts_dir = ctx.run_dir / "artifacts"
        X_scaled_with_meta = X_scaled.copy()
        X_scaled_with_meta["split"] = joined["split"].values
        X_scaled_with_meta["label_binary"] = joined["label_binary_raw"].values

        out_path = artifacts_dir / "kd05_features_scaled.parquet"
        X_scaled_with_meta.to_parquet(out_path, index=True)

        scaler_path = artifacts_dir / "kd05_scaler.joblib"
        joblib.dump(scaler, scaler_path)

        return {
            "features_parquet": str(out_path),
            "scaler_joblib": str(scaler_path),
            "n_feature_cols": len(feature_cols),
            "n_count_like_logged": len(count_like_cols),
            "n_rows": len(X_scaled_with_meta),
        }
