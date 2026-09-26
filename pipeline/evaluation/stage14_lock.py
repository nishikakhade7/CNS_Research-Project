"""Stage 14 — Lock Final Pipeline Configuration.

GROUP 3 SCOPE: LightGBM only. Selects the best-performing modality BY macro-F1 among the
results Stage 13 actually produced across the full modality experiment matrix — never
declared before that matrix was run. Re-fits that one winning (modality, LightGBM)
combination deterministically (same seed) so a concrete, checksummed artifact exists to
freeze, and records the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import joblib
import pandas as pd

from pipeline.adapters import lightgbm_adapter
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.selection.stage08_selection import MODALITY_ARTIFACTS, _get_labels

log = get_stage_logger("stage14_lock")

DEFAULT_TREE_K = 270


class LockStage(Stage):
    name = "stage14_lock"
    depends_on = ["stage13_ablation"]

    def config_section(self, cfg: dict) -> dict:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        with open(ctx.run_dir / "metrics" / "stage13_ablation_results.json", encoding="utf-8") as f:
            ablation = json.load(f)

        best = None
        for modality, models in ablation.items():
            metrics = models["lightgbm"]
            if best is None or metrics["macro_f1"] > best["macro_f1"]:
                best = {"modality": modality, "macro_f1": metrics["macro_f1"], "feature_count": metrics["feature_count"]}

        if best is None:
            raise RuntimeError("Stage 14 hard stop: no LightGBM metrics found in Stage 13 ablation results.")

        log.info(f"Best modality by macro-F1: {best['modality']} macro_f1={best['macro_f1']:.4f} "
                 f"({best['feature_count']} features)")

        cfg = ctx.cfg
        env = capture_environment()
        gpu_verified = env["lightgbm_gpu"]["supported"]

        modality = best["modality"]
        path = ctx.run_dir / "artifacts" / MODALITY_ARTIFACTS[modality]
        df = pd.read_parquet(path)
        y_all = _get_labels(df, ctx.run_dir)

        with open(ctx.run_dir / "artifacts" / "stage08_feature_selection.json", encoding="utf-8") as f:
            sel = json.load(f)
        k_available = max(sel[modality]["methods"]["lightgbm_importance"].keys(), key=lambda k: int(k))
        ranked = sel[modality]["methods"]["lightgbm_importance"][k_available]

        train_mask = (df["split"] == "train").to_numpy()
        y_train = y_all[train_mask].to_numpy()

        cols = ranked[:best["feature_count"]]
        model, device = lightgbm_adapter.build_model(cfg, gpu_verified)

        t0 = time.time()
        model.fit(df.loc[train_mask, cols].to_numpy(), y_train)
        train_time = time.time() - t0

        frozen_path = ctx.run_dir / "artifacts" / "stage14_frozen_model.joblib"
        joblib.dump({"model": model, "modality": modality, "model_name": "lightgbm", "feature_columns": cols}, frozen_path)

        checksum = hashlib.sha256(open(frozen_path, "rb").read()).hexdigest()[:16]

        lock_manifest = {
            "selected_model": "lightgbm",
            "selected_modality": modality,
            "feature_columns": cols,
            "n_features": len(cols),
            "internal_macro_f1_at_selection_time": best["macro_f1"],
            "retrain_time_sec": round(train_time, 3),
            "frozen_artifact": str(frozen_path),
            "frozen_artifact_sha256_prefix": checksum,
            "config_snapshot": cfg,
        }
        lock_path = ctx.run_dir / "artifacts" / "stage14_lock_manifest.json"
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_manifest, f, indent=2, default=str)

        log.info(f"LOCKED: lightgbm/{modality}, {len(cols)} features, checksum={checksum} -> {frozen_path}")
        return {"lock_manifest": str(lock_path), "frozen_model": str(frozen_path),
                "selected_model": "lightgbm", "selected_modality": modality}
