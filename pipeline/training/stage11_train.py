"""Stage 11 — Model Training (all 5 model families, primary modality = dynamic_470).

Trains QDA, NuSVC, LightGBM, Balanced Random Forest, and the Autoencoder on the
dynamic-470 modality (the primary view — Doc 05) using Stage 08's train-only feature
ranking and Stage 09's train-only imbalance handling. Stage 13 repeats this same logic
across all four modalities for the ablation comparison; this stage exists to establish and
checkpoint a single correct, fully-evaluated baseline first.

Every model records: training time, inference time (on the val split), feature count, and
a full configuration snapshot — nothing is trained "silently."
"""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from pipeline.adapters import balanced_rf, lightgbm_adapter, nusvc, qda
from pipeline.adapters.autoencoder import encode, train_autoencoder
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage11_train")

PRIMARY_MODALITY = "dynamic_470"
PRIMARY_ARTIFACT = "stage06_dynamic_470_features.parquet"
DEFAULT_TREE_K = 270


def _load_ranked_features(ctx: StageContext, modality: str) -> list[str]:
    with open(ctx.run_dir / "artifacts" / "stage08_feature_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    ranking = sel[modality]["methods"]["lightgbm_importance"][str(DEFAULT_TREE_K)]
    return ranking


class TrainingStage(Stage):
    name = "stage11_train"
    depends_on = ["stage08_selection", "stage09_imbalance"]

    def config_section(self, cfg: dict) -> dict:
        return {"models": cfg["models"], "imbalance": cfg["imbalance"], "seed": cfg["seed"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        env = capture_environment()
        gpu_verified = env["lightgbm_gpu"]["supported"]

        df = pd.read_parquet(ctx.run_dir / "artifacts" / PRIMARY_ARTIFACT)
        ranked = _load_ranked_features(ctx, PRIMARY_MODALITY)

        train_mask = (df["split"] == "train").to_numpy()
        val_mask = (df["split"] == "val").to_numpy()
        y_train_full = df.loc[train_mask, "Class"].to_numpy()
        y_val = df.loc[val_mask, "Class"].to_numpy()

        with open(ctx.run_dir / "artifacts" / "stage09_imbalance_report.json", encoding="utf-8") as f:
            imb = json.load(f)[PRIMARY_MODALITY]
        class_weight = {int(k): v for k, v in imb["class_weights"].items()}

        results: dict[str, Any] = {}
        model_dir = ctx.run_dir / "checkpoints"

        # ---- LightGBM ----
        raw_cols = lightgbm_adapter.select_view(ranked, DEFAULT_TREE_K)
        X_train, X_val = df.loc[train_mask, raw_cols].to_numpy(), df.loc[val_mask, raw_cols].to_numpy()
        model, device_used = lightgbm_adapter.build_model(cfg, gpu_verified)
        t0 = time.time()
        model.fit(X_train, y_train_full, sample_weight=np.array([class_weight[y] for y in y_train_full]))
        train_time = time.time() - t0
        t0 = time.time()
        preds = model.predict(X_val)
        infer_time = time.time() - t0
        joblib.dump(model, model_dir / "lightgbm_dynamic470.joblib")
        results["lightgbm"] = {
            "device_used": device_used, "feature_count": len(raw_cols),
            "train_time_sec": round(train_time, 3), "inference_time_sec": round(infer_time, 4),
            "val_accuracy": float((preds == y_val).mean()),
        }
        log.info(f"LightGBM trained on {device_used}: val_accuracy={results['lightgbm']['val_accuracy']:.4f} "
                 f"({len(raw_cols)} features, {train_time:.1f}s train)")

        # ---- Balanced Random Forest ----
        model = balanced_rf.build_model(cfg)
        t0 = time.time()
        model.fit(X_train, y_train_full)
        train_time = time.time() - t0
        t0 = time.time()
        preds = model.predict(X_val)
        infer_time = time.time() - t0
        joblib.dump(model, model_dir / "balanced_rf_dynamic470.joblib")
        results["balanced_rf"] = {
            "feature_count": len(raw_cols), "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 4), "val_accuracy": float((preds == y_val).mean()),
        }
        log.info(f"Balanced RF: val_accuracy={results['balanced_rf']['val_accuracy']:.4f}")

        # ---- QDA ----
        scaled_cols = qda.select_view(ranked, ranked, cfg)
        Xs_train, Xs_val = df.loc[train_mask, scaled_cols].to_numpy(), df.loc[val_mask, scaled_cols].to_numpy()
        model = qda.build_model(cfg)
        t0 = time.time()
        model.fit(Xs_train, y_train_full)
        train_time = time.time() - t0
        t0 = time.time()
        preds = model.predict(Xs_val)
        infer_time = time.time() - t0
        joblib.dump(model, model_dir / "qda_dynamic470.joblib")
        results["qda"] = {
            "feature_count": len(scaled_cols), "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 4), "val_accuracy": float((preds == y_val).mean()),
        }
        log.info(f"QDA: val_accuracy={results['qda']['val_accuracy']:.4f} ({len(scaled_cols)} features)")

        # ---- NuSVC ----
        svc_cols = nusvc.select_view(ranked, ranked, cfg)
        Xs_train, Xs_val = df.loc[train_mask, svc_cols].to_numpy(), df.loc[val_mask, svc_cols].to_numpy()
        model = nusvc.build_model(cfg)
        t0 = time.time()
        model.fit(Xs_train, y_train_full)
        train_time = time.time() - t0
        t0 = time.time()
        preds = model.predict(Xs_val)
        infer_time = time.time() - t0
        joblib.dump(model, model_dir / "nusvc_dynamic470.joblib")
        results["nusvc"] = {
            "feature_count": len(svc_cols), "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 4), "val_accuracy": float((preds == y_val).mean()),
        }
        log.info(f"NuSVC: val_accuracy={results['nusvc']['val_accuracy']:.4f}")

        # ---- Autoencoder ----
        ae_cols = [f"{c}__scaled" for c in ranked[:DEFAULT_TREE_K]]
        Xae_train, Xae_val = df.loc[train_mask, ae_cols].to_numpy(dtype=np.float32), df.loc[val_mask, ae_cols].to_numpy(dtype=np.float32)
        t0 = time.time()
        ae_model, device = train_autoencoder(Xae_train, cfg, log)
        train_time = time.time() - t0
        t0 = time.time()
        latent_val, err_val = encode(ae_model, device, Xae_val)
        infer_time = time.time() - t0
        torch.save(ae_model.state_dict(), model_dir / "autoencoder_dynamic470.pt")
        results["autoencoder"] = {
            "feature_count": len(ae_cols), "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 4), "device": str(device),
            "val_mean_reconstruction_error": float(err_val.mean()),
            "latent_dim": cfg["models"]["autoencoder"]["latent_dim"],
        }
        log.info(f"Autoencoder trained on {device}: val_mean_reconstruction_error={err_val.mean():.5f}")

        out_path = ctx.run_dir / "metrics" / "stage11_training_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {"training_results": str(out_path), "models_trained": list(results.keys()), "modality": PRIMARY_MODALITY}
