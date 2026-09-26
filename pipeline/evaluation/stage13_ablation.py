"""Stage 13 — Ablation / Modality Comparison.

Runs all 5 models across all 4 modality experiments (static-only, dynamic-139, dynamic-470,
fused) required by the project brief, using each modality's own train-only feature ranking
(Stage 08) and reporting TEST-split metrics for a direct, fair comparison. This is what
actually determines whether the 470-feature view "helps" or whether fusion "helps" —
neither is assumed going in.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from pipeline.adapters import balanced_rf, lightgbm_adapter, nusvc, qda
from pipeline.adapters.autoencoder import encode, train_autoencoder
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report
from pipeline.selection.stage08_selection import MODALITY_ARTIFACTS, _get_labels

log = get_stage_logger("stage13_ablation")

DEFAULT_TREE_K = 270


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

    def scaled_name(c: str) -> str:
        return f"{c}__scaled" if f"{c}__scaled" in df.columns else c  # fused branch already prefixes; guard just in case

    results: dict[str, Any] = {}

    # LightGBM
    raw_cols = ranked[:DEFAULT_TREE_K]
    model, device = lightgbm_adapter.build_model(cfg, gpu_verified)
    t0 = time.time()
    model.fit(df.loc[train_mask, raw_cols].to_numpy(), y_train)
    train_time = time.time() - t0
    preds = model.predict(df.loc[test_mask, raw_cols].to_numpy())
    results["lightgbm"] = {**full_classification_report(y_test, preds, labels),
                            "feature_count": len(raw_cols), "train_time_sec": round(train_time, 3), "device": device}

    # Balanced RF
    model = balanced_rf.build_model(cfg)
    t0 = time.time()
    model.fit(df.loc[train_mask, raw_cols].to_numpy(), y_train)
    train_time = time.time() - t0
    preds = model.predict(df.loc[test_mask, raw_cols].to_numpy())
    results["balanced_rf"] = {**full_classification_report(y_test, preds, labels),
                               "feature_count": len(raw_cols), "train_time_sec": round(train_time, 3)}

    # QDA (capped, scaled)
    qda_cols = [scaled_name(c) for c in ranked[: cfg["models"]["qda"]["max_features"]]]
    model = qda.build_model(cfg)
    t0 = time.time()
    model.fit(df.loc[train_mask, qda_cols].to_numpy(), y_train)
    train_time = time.time() - t0
    preds = model.predict(df.loc[test_mask, qda_cols].to_numpy())
    results["qda"] = {**full_classification_report(y_test, preds, labels),
                       "feature_count": len(qda_cols), "train_time_sec": round(train_time, 3)}

    # NuSVC (capped, scaled)
    svc_cols = [scaled_name(c) for c in ranked[: cfg["models"]["nusvc"]["max_features"]]]
    model = nusvc.build_model(cfg)
    t0 = time.time()
    model.fit(df.loc[train_mask, svc_cols].to_numpy(), y_train)
    train_time = time.time() - t0
    preds = model.predict(df.loc[test_mask, svc_cols].to_numpy())
    results["nusvc"] = {**full_classification_report(y_test, preds, labels),
                         "feature_count": len(svc_cols), "train_time_sec": round(train_time, 3)}

    # Autoencoder (unsupervised reconstruction error only)
    ae_cols = [scaled_name(c) for c in ranked[:DEFAULT_TREE_K]]
    Xae_train = df.loc[train_mask, ae_cols].to_numpy(dtype=np.float32)
    Xae_test = df.loc[test_mask, ae_cols].to_numpy(dtype=np.float32)
    ae_model, device_t = train_autoencoder(Xae_train, cfg, log)
    _, err = encode(ae_model, device_t, Xae_test)
    results["autoencoder"] = {"mean_reconstruction_error": float(err.mean()), "feature_count": len(ae_cols)}

    log.info(f"[{modality}] lightgbm macro_f1={results['lightgbm']['macro_f1']:.4f} | "
             f"balanced_rf macro_f1={results['balanced_rf']['macro_f1']:.4f} | "
             f"qda macro_f1={results['qda']['macro_f1']:.4f} | nusvc macro_f1={results['nusvc']['macro_f1']:.4f}")
    return results


class AblationStage(Stage):
    name = "stage13_ablation"
    depends_on = ["stage08_selection", "stage09_imbalance"]

    def config_section(self, cfg: dict) -> dict:
        return {"modalities": cfg["modalities"], "models": cfg["models"]}

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
