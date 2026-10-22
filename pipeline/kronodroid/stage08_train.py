"""KronoDroid Stage 08 — LightGBM training (binary malware/benign).

Trains TWO models, both on TRAIN split only, evaluated later (Stage 09) on the untouched
val/test splits:

  1. `replication` — top-50 features selected by ExtraTrees importance (Stage 06),
     MinMax-scaled: the closest reproduction of Waheed & Qadir (2024)'s best-reported
     config, but with LightGBM instead of their Random Forest (their paper didn't use
     LightGBM; this is OUR model on THEIR feature-selection recipe, not their numbers).
  2. `full_standardized` — all features, StandardScaler-scaled (Stage 05's default
     representation): this project's own choice, consistent with the MalDroid pipeline's
     existing LightGBM stage (pipeline/training/stage11_train.py), kept separate so the
     comparison in Stage 09 is explicit about which config produced which number.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pipeline.core.stage import Stage, StageContext


class KronoDroidTrainStage(Stage):
    name = "kd08_train"
    depends_on = ["kd06_selection", "kd07_imbalance"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"models": cfg["models"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        df = pd.read_parquet(ctx.run_dir / "artifacts" / "kd05_features_scaled.parquet")
        feature_cols = [c for c in df.columns if c not in ("split", "label_binary")]

        with open(ctx.run_dir / "artifacts" / "kd06_feature_rankings.json", encoding="utf-8") as f:
            rankings = json.load(f)
        top50 = rankings["top_k_sets"]["et_top50"]

        train = df[df["split"] == "train"]
        val = df[df["split"] == "val"]

        lgbm_cfg = dict(cfg["models"]["lightgbm"])
        lgbm_cfg.pop("class_weight", None)

        checkpoints_dir = ctx.run_dir / "checkpoints"
        results: dict[str, Any] = {}

        # --- Model 1: replication config (top-50 ET features, MinMax scaled) ---
        mm = MinMaxScaler()
        X_train_rep = mm.fit_transform(train[top50])
        X_val_rep = mm.transform(val[top50])
        model_rep = lgb.LGBMClassifier(class_weight="balanced", verbosity=-1, **lgbm_cfg)
        model_rep.fit(
            X_train_rep, train["label_binary"],
            eval_set=[(X_val_rep, val["label_binary"])],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        rep_path = checkpoints_dir / "kd08_model_replication.joblib"
        joblib.dump({"model": model_rep, "scaler": mm, "feature_cols": top50}, rep_path)
        results["replication"] = {
            "model_path": str(rep_path),
            "feature_cols": top50,
            "n_features": len(top50),
            "best_iteration": int(model_rep.best_iteration_),
        }

        # --- Model 2: full feature set, StandardScaler (already applied in Stage 05) ---
        X_train_full = train[feature_cols]
        X_val_full = val[feature_cols]
        model_full = lgb.LGBMClassifier(class_weight="balanced", verbosity=-1, **lgbm_cfg)
        model_full.fit(
            X_train_full, train["label_binary"],
            eval_set=[(X_val_full, val["label_binary"])],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        full_path = checkpoints_dir / "kd08_model_full_standardized.joblib"
        joblib.dump({"model": model_full, "scaler": None, "feature_cols": feature_cols}, full_path)
        results["full_standardized"] = {
            "model_path": str(full_path),
            "n_features": len(feature_cols),
            "best_iteration": int(model_full.best_iteration_),
        }

        return results
