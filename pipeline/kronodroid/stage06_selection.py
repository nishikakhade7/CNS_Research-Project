"""KronoDroid Stage 06 — feature selection.

Replicates Waheed & Qadir (2024)'s filter-method approach as a baseline: ExtraTreesClassifier
importance + mutual information, with MinMax scaling (their best-reported config), compared
across top-30/50/100 feature counts (their tested sizes). This stage only RANKS and records
the comparison; Stage 08 (training) decides which top-K to actually train the final model on.

This is explicitly "replicating Waheed & Qadir" — it does not add this project's own
multi-method selection layer (that would be a further stage, not yet built, same honesty
rule as the task brief: label what's replication vs. our own addition).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from pipeline.core.stage import Stage, StageContext


class KronoDroidSelectionStage(Stage):
    name = "kd06_selection"
    depends_on = ["kd05_preprocess"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"feature_selection": cfg["feature_selection"], "seed": cfg["seed"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        df = pd.read_parquet(ctx.run_dir / "artifacts" / "kd05_features_scaled.parquet")
        feature_cols = [c for c in df.columns if c not in ("split", "label_binary")]

        train = df[df["split"] == "train"]
        X_train = train[feature_cols].to_numpy()
        y_train = train["label_binary"].to_numpy()

        # Waheed & Qadir's best config used MinMax scaling specifically for selection;
        # Stage 05's StandardScaler output is re-scaled to [0,1] here for this comparison,
        # not reused as the final training representation.
        mm = MinMaxScaler()
        X_mm = mm.fit_transform(X_train)

        seed = cfg["seed"]
        et = ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        et.fit(X_mm, y_train)
        et_importance = pd.Series(et.feature_importances_, index=feature_cols).sort_values(ascending=False)

        mi = mutual_info_classif(X_mm, y_train, random_state=seed, n_jobs=-1)
        mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)

        top_k_candidates = cfg["feature_selection"]["top_k_candidates"]
        rankings = {
            "extra_trees_importance": et_importance.index.tolist(),
            "mutual_info": mi_series.index.tolist(),
        }
        top_k_sets = {
            f"et_top{k}": et_importance.head(k).index.tolist() for k in top_k_candidates
        } | {
            f"mi_top{k}": mi_series.head(k).index.tolist() for k in top_k_candidates
        }

        out_path = ctx.run_dir / "artifacts" / "kd06_feature_rankings.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"rankings": rankings, "top_k_sets": top_k_sets}, f, indent=2)

        overlap_at_50 = len(set(top_k_sets["et_top50"]) & set(top_k_sets["mi_top50"]))

        return {
            "rankings_json": str(out_path),
            "n_features_ranked": len(feature_cols),
            "top_k_candidates": top_k_candidates,
            "et_mi_overlap_at_top50": overlap_at_50,
        }
