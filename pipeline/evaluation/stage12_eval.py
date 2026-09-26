"""Stage 12 — Internal Evaluation (MalDroid TEST split, dynamic_470, all 5 models).

Test split is used here for the first and only time in the internal pipeline — val was
used during Stage 11's quick checks. Because accuracy alone is misleading under class
imbalance (3.1x on MalDroid), macro-F1 is the headline metric, reported alongside
everything else per Doc 06 Stage 12 requirements.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from pipeline.adapters import balanced_rf, lightgbm_adapter, nusvc, qda  # noqa: F401 (adapters define select_view used below via training stage's saved feature lists)
from pipeline.adapters.autoencoder import Autoencoder, encode
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report
from pipeline.training.stage11_train import DEFAULT_TREE_K, PRIMARY_ARTIFACT, PRIMARY_MODALITY, _load_ranked_features

log = get_stage_logger("stage12_eval")


class InternalEvaluationStage(Stage):
    name = "stage12_eval"
    depends_on = ["stage11_train"]

    def config_section(self, cfg: dict) -> dict:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        df = pd.read_parquet(ctx.run_dir / "artifacts" / PRIMARY_ARTIFACT)
        ranked = _load_ranked_features(ctx, PRIMARY_MODALITY)
        test_mask = (df["split"] == "test").to_numpy()
        y_test = df.loc[test_mask, "Class"].to_numpy()
        labels = sorted(df["Class"].unique().tolist())

        model_dir = ctx.run_dir / "checkpoints"
        results: dict[str, Any] = {}

        raw_cols = ranked[:DEFAULT_TREE_K]
        scaled_cols = [f"{c}__scaled" for c in ranked[: cfg["models"]["qda"]["max_features"]]]
        svc_cols = [f"{c}__scaled" for c in ranked[: cfg["models"]["nusvc"]["max_features"]]]

        for name, cols, path in [
            ("lightgbm", raw_cols, model_dir / "lightgbm_dynamic470.joblib"),
            ("balanced_rf", raw_cols, model_dir / "balanced_rf_dynamic470.joblib"),
            ("qda", scaled_cols, model_dir / "qda_dynamic470.joblib"),
            ("nusvc", svc_cols, model_dir / "nusvc_dynamic470.joblib"),
        ]:
            model = joblib.load(path)
            X_test = df.loc[test_mask, cols].to_numpy()
            preds = model.predict(X_test)
            report = full_classification_report(y_test, preds, labels)
            results[name] = report
            log.info(f"[{name}] TEST macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f} "
                     f"balanced_accuracy={report['balanced_accuracy']:.4f}")

        # Autoencoder: report reconstruction error by class (unsupervised signal, not a classifier)
        ae_cols = [f"{c}__scaled" for c in ranked[:DEFAULT_TREE_K]]
        ae_state = torch.load(model_dir / "autoencoder_dynamic470.pt", map_location="cpu")
        model = Autoencoder(len(ae_cols), cfg["models"]["autoencoder"]["hidden_dims"], cfg["models"]["autoencoder"]["latent_dim"])
        model.load_state_dict(ae_state)
        model.eval()
        X_test_ae = df.loc[test_mask, ae_cols].to_numpy(dtype=np.float32)
        _, err = encode(model, torch.device("cpu"), X_test_ae)
        per_class_err = {}
        for c in labels:
            mask_c = y_test == c
            per_class_err[str(c)] = float(err[mask_c].mean()) if mask_c.any() else None
        results["autoencoder"] = {"mean_reconstruction_error": float(err.mean()), "per_class_mean_reconstruction_error": per_class_err}
        log.info(f"[autoencoder] TEST mean_reconstruction_error={err.mean():.5f}")

        out_path = ctx.run_dir / "metrics" / "stage12_internal_evaluation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {"internal_evaluation": str(out_path), "models_evaluated": list(results.keys())}
