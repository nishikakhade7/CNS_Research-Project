"""Stage 13 — Ablation / Modality Comparison.

GROUP 3 SCOPE: LightGBM only. Runs it across all 4 modality experiments required by the
project brief (static-only, dynamic-139, dynamic-470, fused), using each modality's own
train-only feature ranking (Stage 08), reporting TEST-split metrics for a direct, fair
comparison. This is what actually determines whether the 470-feature view "helps" or
whether fusion "helps" for Group 3's model — neither is assumed going in.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from pipeline.adapters import lightgbm_adapter
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report
from pipeline.selection.stage08_selection import MODALITY_ARTIFACTS, _get_labels

log = get_stage_logger("stage13_ablation")

DEFAULT_TREE_K = 270


def load_class_weights(ctx: StageContext, modality: str) -> dict[str, float]:
    """Load Stage 09's train-only-computed class weights for a modality. Shared by
    Stage 13 and Stage 14 so both apply the SAME weighting Stage 11 already uses —
    previously Stage 13/14 silently trained unweighted despite config.imbalance.strategy
    being 'class_weight', found during forensic audit of the reported ~98% result."""
    with open(ctx.run_dir / "artifacts" / "stage09_imbalance_report.json", encoding="utf-8") as f:
        imb = json.load(f)[modality]
    return dict(imb["class_weights"].items())


def _run_modality(modality: str, cfg: dict, ctx: StageContext, gpu_verified: bool) -> dict[str, Any]:
    path = ctx.run_dir / "artifacts" / MODALITY_ARTIFACTS[modality]
    df = pd.read_parquet(path)
    y_all = _get_labels(df, ctx.run_dir)
    labels = sorted(pd.Series(y_all).dropna().unique().tolist())

    with open(ctx.run_dir / "artifacts" / "stage08_feature_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    k_tree = str(min(DEFAULT_TREE_K, sel[modality]["n_raw_features"]))
    if k_tree not in sel[modality]["methods"]["lightgbm_importance"]:
        k_tree = max(sel[modality]["methods"]["lightgbm_importance"].keys(), key=lambda k: int(k))
    ranked = sel[modality]["methods"]["lightgbm_importance"][k_tree]

    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train = y_all[train_mask].to_numpy()
    y_test = y_all[test_mask].to_numpy()

    apply_weights = cfg["imbalance"]["strategy"] == "class_weight"
    sample_weight = None
    if apply_weights:
        class_weight = load_class_weights(ctx, modality)
        sample_weight = np.array([class_weight[y] for y in y_train])

    raw_cols = ranked[: min(DEFAULT_TREE_K, len(ranked))]
    model, device = lightgbm_adapter.build_model(cfg, gpu_verified)
    t0 = time.time()
    model.fit(df.loc[train_mask, raw_cols].to_numpy(), y_train, sample_weight=sample_weight)
    train_time = time.time() - t0
    preds = model.predict(df.loc[test_mask, raw_cols].to_numpy())
    report = {**full_classification_report(y_test, preds, labels),
              "feature_count": len(raw_cols), "train_time_sec": round(train_time, 3), "device": device,
              "class_weight_applied": apply_weights}

    log.info(f"[{modality}] lightgbm macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f} "
             f"(class_weight_applied={apply_weights})")
    return {"lightgbm": report}


class AblationStage(Stage):
    name = "stage13_ablation"
    depends_on = ["stage08_selection", "stage09_imbalance"]

    def config_section(self, cfg: dict) -> dict:
        return {"modalities": cfg["modalities"], "models": {"lightgbm": cfg["models"]["lightgbm"]},
                "imbalance": cfg["imbalance"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        env = capture_environment()
        gpu_verified = env["lightgbm_gpu"]["supported"]

        all_results: dict[str, Any] = {}
        for modality in cfg["modalities"]:
            log.info(f"=== Ablation: modality '{modality}' ===")
            all_results[modality] = _run_modality(modality, cfg, ctx, gpu_verified)

        out_path = ctx.run_dir / "metrics" / "stage13_ablation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        log.info(f"Ablation complete across {list(all_results.keys())} -> {out_path}")
        return {"ablation_results": str(out_path), "modalities": list(all_results.keys())}
