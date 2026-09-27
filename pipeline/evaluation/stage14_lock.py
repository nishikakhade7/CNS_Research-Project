"""Stage 14 — Lock Final Pipeline Configuration.

GROUP 3 SCOPE: LightGBM only. Selects the best-performing modality BY macro-F1 among the
results Stage 13 actually produced across the full modality experiment matrix — never
declared before that matrix was run. Re-fits that one winning (modality, LightGBM)
combination deterministically (same seed, same class weighting as Stage 13) so a concrete,
checksummed artifact exists to freeze, and records the exact configuration that produced it.

Unlike an earlier version of this stage, it does NOT simply trust Stage 13's reported
metric — it independently re-evaluates the freshly-retrained model on the TEST split and
persists its own `verified_test_metrics`, flagging (not silently accepting) any mismatch
against Stage 13's number. Found during forensic audit: the earlier version relied on an
unverified assumption that the retrain would be numerically identical to Stage 13's run.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.adapters import lightgbm_adapter
from pipeline.core.environment import capture_environment
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report
from pipeline.evaluation.stage13_ablation import load_class_weights
from pipeline.selection.stage08_selection import MODALITY_ARTIFACTS, _get_labels

log = get_stage_logger("stage14_lock")

DEFAULT_TREE_K = 270
METRIC_MISMATCH_TOLERANCE = 0.01  # macro-F1 absolute difference before flagging as a discrepancy


class LockStage(Stage):
    name = "stage14_lock"
    depends_on = ["stage13_ablation"]

    def config_section(self, cfg: dict) -> dict:
        return {"imbalance": cfg["imbalance"]}

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
        labels = sorted(pd.Series(y_all).dropna().unique().tolist())

        with open(ctx.run_dir / "artifacts" / "stage08_feature_selection.json", encoding="utf-8") as f:
            sel = json.load(f)
        k_available = max(sel[modality]["methods"]["lightgbm_importance"].keys(), key=lambda k: int(k))
        ranked = sel[modality]["methods"]["lightgbm_importance"][k_available]

        train_mask = (df["split"] == "train").to_numpy()
        test_mask = (df["split"] == "test").to_numpy()
        y_train = y_all[train_mask].to_numpy()
        y_test = y_all[test_mask].to_numpy()

        # Apply the SAME class weighting Stage 13 used for this modality — an earlier
        # version trained unweighted here regardless of config.imbalance.strategy.
        apply_weights = cfg["imbalance"]["strategy"] == "class_weight"
        sample_weight = None
        if apply_weights:
            class_weight = load_class_weights(ctx, modality)
            sample_weight = np.array([class_weight[y] for y in y_train])

        cols = ranked[:best["feature_count"]]
        model, device = lightgbm_adapter.build_model(cfg, gpu_verified)

        t0 = time.time()
        model.fit(df.loc[train_mask, cols].to_numpy(), y_train, sample_weight=sample_weight)
        train_time = time.time() - t0

        # Independently re-evaluate the freshly-retrained model on TEST — do not just
        # trust Stage 13's number for a DIFFERENT (freshly re-fit) model object.
        preds = model.predict(df.loc[test_mask, cols].to_numpy())
        verified_metrics = full_classification_report(y_test, preds, labels)
        macro_f1_diff = abs(verified_metrics["macro_f1"] - best["macro_f1"])
        metrics_match = macro_f1_diff <= METRIC_MISMATCH_TOLERANCE
        if not metrics_match:
            log.warning(
                f"DISCREPANCY: Stage 14's independently re-evaluated macro_f1="
                f"{verified_metrics['macro_f1']:.4f} differs from Stage 13's reported "
                f"{best['macro_f1']:.4f} by {macro_f1_diff:.4f} (tolerance={METRIC_MISMATCH_TOLERANCE}). "
                f"This means the retrain was NOT reproducible — investigate before trusting the locked model."
            )
        else:
            log.info(f"Verified: Stage 14 retrain macro_f1={verified_metrics['macro_f1']:.4f} matches "
                      f"Stage 13's reported {best['macro_f1']:.4f} within tolerance.")

        frozen_path = ctx.run_dir / "artifacts" / "stage14_frozen_model.joblib"
        joblib.dump({"model": model, "modality": modality, "model_name": "lightgbm", "feature_columns": cols}, frozen_path)

        checksum = hashlib.sha256(open(frozen_path, "rb").read()).hexdigest()[:16]

        lock_manifest = {
            "selected_model": "lightgbm",
            "selected_modality": modality,
            "feature_columns": cols,
            "n_features": len(cols),
            "class_weight_applied": apply_weights,
            "internal_macro_f1_at_selection_time": best["macro_f1"],
            "verified_test_metrics": verified_metrics,
            "metrics_match_stage13": metrics_match,
            "metrics_match_tolerance": METRIC_MISMATCH_TOLERANCE,
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
                "selected_model": "lightgbm", "selected_modality": modality,
                "metrics_match_stage13": metrics_match}
