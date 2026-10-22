"""KronoDroid Stage 09 — internal evaluation on the held-out TEST split (touched for the
first time here), plus an explicit comparison against Waheed & Qadir (2024)'s published
Random Forest numbers on the same real_device data.

The comparison is NOT apples-to-apples and this stage's output says so directly: different
model (LightGBM vs. their Random Forest), different split protocol (fixed 70/15/15 vs.
their 70/30 + 5-fold CV on the 70%), and our own early-stopping/val-based model selection.
Same intellectual-honesty rule as Docs/final/03_EXTERNAL_VALIDATION_ANDMAL.md.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from pipeline.core.stage import Stage, StageContext

# Waheed & Qadir (2024), Security and Communication Networks — binary detection, their
# best-reported config (RF, top-50 ET features, MinMax scaling). CITED, not measured by us.
CITED_BENCHMARK_BINARY = {
    "source": "Waheed & Qadir (2024), Security and Communication Networks",
    "model": "Random Forest (GridSearchCV-tuned)",
    "accuracy": 0.9803,
    "precision": 0.9851,
    "recall": 0.9773,
    "f1": 0.9812,
}


class KronoDroidEvalStage(Stage):
    name = "kd09_eval"
    depends_on = ["kd08_train"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        df = pd.read_parquet(ctx.run_dir / "artifacts" / "kd05_features_scaled.parquet")
        test = df[df["split"] == "test"]
        y_test = test["label_binary"]

        checkpoints_dir = ctx.run_dir / "checkpoints"
        results: dict[str, Any] = {"cited_benchmark_binary": CITED_BENCHMARK_BINARY}

        for key, fname in [
            ("replication", "kd08_model_replication.joblib"),
            ("full_standardized", "kd08_model_full_standardized.joblib"),
        ]:
            bundle = joblib.load(checkpoints_dir / fname)
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]
            scaler = bundle["scaler"]

            X_test = test[feature_cols]
            if scaler is not None:
                X_test = scaler.transform(X_test)

            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred)), 4),
                "recall": round(float(recall_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred)), 4),
                "n_test_samples": int(len(y_test)),
            }
            results[key] = metrics

        out_path = ctx.run_dir / "metrics" / "kd09_eval_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {"metrics_json": str(out_path), **results}
