"""Stage 08 — Feature Selection (TRAIN ONLY).

Modular selector supporting mutual information, chi-square (only where features are
non-negative, which our log1p-transformed counts and binary flags are), and LightGBM
importance. ReliefF is intentionally NOT implemented in this pass — no vetted ReliefF
package (e.g. skrebate) is installed, and this stage does not fabricate a substitute; it is
recorded here and in the final report as a documented gap, not silently skipped.

No K value is claimed optimal — all configured K values are run and reported side by side;
Stage 13 (ablation) is what compares them empirically.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif

from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage08_selection")

MODALITY_ARTIFACTS = {
    "static": "stage05_static_features.parquet",
    "dynamic_139": "stage06_dynamic_139_features.parquet",
    "dynamic_470": "stage06_dynamic_470_features.parquet",
    "fused": "stage07_fused_features.parquet",
}


def _raw_feature_columns(df: pd.DataFrame, modality: str) -> list[str]:
    if modality == "fused":
        return [c for c in df.columns if "__scaled" not in c and c not in ("sample_id", "split", "Class", "label_canonical")]
    return [c for c in df.columns if "__scaled" not in c and c not in ("sample_id", "split", "Class")]


def _get_labels(df: pd.DataFrame, run_dir) -> pd.Series:
    """Always returns the canonical string label (e.g. 'Adware'), never the raw numeric
    Class code — even for modalities whose parquet happens to carry a native 'Class'
    column — so every modality's confusion matrices/reports use the same label space and
    are directly comparable in Stage 13's ablation table."""
    split_manifest = pd.read_parquet(run_dir / "manifests" / "maldroid_split_manifest.parquet")
    merged = df[["sample_id"]].merge(split_manifest[["sample_id", "label_canonical"]], on="sample_id", how="left")
    return merged["label_canonical"]


class FeatureSelectionStage(Stage):
    name = "stage08_selection"
    depends_on = ["stage05_static", "stage06_dynamic", "stage07_fusion"]

    def config_section(self, cfg: dict) -> dict:
        return {"feature_selection": cfg["feature_selection"], "modalities": cfg["modalities"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        fs_cfg = cfg["feature_selection"]
        methods, k_values = fs_cfg["methods"], fs_cfg["k_values"]

        results: dict[str, Any] = {}
        for modality in cfg["modalities"]:
            path = ctx.run_dir / "artifacts" / MODALITY_ARTIFACTS[modality]
            df = pd.read_parquet(path)
            feat_cols = _raw_feature_columns(df, modality)
            y_all = _get_labels(df, ctx.run_dir)
            train_mask = (df["split"] == "train").to_numpy()

            X_train = df.loc[train_mask, feat_cols].to_numpy(dtype=float)
            y_train = y_all[train_mask].to_numpy()

            log.info(f"[{modality}] selecting from {len(feat_cols)} raw features, {X_train.shape[0]} train rows")

            modality_result: dict[str, Any] = {"n_raw_features": len(feat_cols), "methods": {}}

            if "mutual_info" in methods:
                mi_scores = mutual_info_classif(X_train, y_train, random_state=cfg["seed"])
                ranked = sorted(zip(feat_cols, mi_scores), key=lambda t: -t[1])
                modality_result["methods"]["mutual_info"] = {
                    str(k): [f for f, _ in ranked[:k]] for k in k_values
                }
                log.info(f"[{modality}] mutual_info top feature: {ranked[0][0]} (score={ranked[0][1]:.4f})")

            if "chi2" in methods:
                if (X_train < 0).any():
                    log.warning(f"[{modality}] chi2 skipped: raw feature matrix has negative values "
                                f"(chi2 requires non-negative input) — documented gap, not fabricated.")
                else:
                    chi2_scores, _ = chi2(X_train, y_train)
                    chi2_scores = np.nan_to_num(chi2_scores, nan=-1.0)
                    ranked = sorted(zip(feat_cols, chi2_scores), key=lambda t: -t[1])
                    modality_result["methods"]["chi2"] = {
                        str(k): [f for f, _ in ranked[:k]] for k in k_values
                    }

            if "lightgbm_importance" in methods:
                try:
                    import lightgbm as lgb
                    clf = lgb.LGBMClassifier(n_estimators=200, random_state=cfg["seed"], verbose=-1)
                    clf.fit(X_train, y_train)
                    ranked = sorted(zip(feat_cols, clf.feature_importances_), key=lambda t: -t[1])
                    modality_result["methods"]["lightgbm_importance"] = {
                        str(k): [f for f, _ in ranked[:k]] for k in k_values
                    }
                    log.info(f"[{modality}] lightgbm_importance top feature: {ranked[0][0]}")
                except ImportError:
                    log.warning(f"[{modality}] lightgbm not installed — skipping lightgbm_importance (documented gap).")

            results[modality] = modality_result

        out_path = ctx.run_dir / "artifacts" / "stage08_feature_selection.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        log.info(f"Feature selection report -> {out_path}")
        return {"selection_report": str(out_path), "modalities": list(results.keys())}
