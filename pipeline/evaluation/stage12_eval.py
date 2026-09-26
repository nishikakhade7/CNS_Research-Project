"""Stage 12 — Internal Evaluation (MalDroid TEST split, dynamic_470).

GROUP 3 SCOPE: evaluates LightGBM only (see Stage 11). Test split is used here for the
first and only time in the internal pipeline — val was used during Stage 11's quick check.
Because accuracy alone is misleading under class imbalance (3.1x on MalDroid), macro-F1 is
the headline metric, reported alongside everything else per Doc 06 Stage 12 requirements.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report
from pipeline.selection.stage08_selection import _get_labels
from pipeline.training.stage11_train import DEFAULT_TREE_K, PRIMARY_ARTIFACT, PRIMARY_MODALITY, _load_ranked_features

log = get_stage_logger("stage12_eval")


class InternalEvaluationStage(Stage):
    name = "stage12_eval"
    depends_on = ["stage11_train"]

    def config_section(self, cfg: dict) -> dict:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        df = pd.read_parquet(ctx.run_dir / "artifacts" / PRIMARY_ARTIFACT)
        ranked = _load_ranked_features(ctx, PRIMARY_MODALITY)
        k = min(DEFAULT_TREE_K, len(ranked))
        # canonical string labels, consistent with Stages 08/09/11/13/14 (an earlier version
        # used the raw numeric Class column, inconsistent with the rest of the pipeline)
        y_all = _get_labels(df, ctx.run_dir)
        test_mask = (df["split"] == "test").to_numpy()
        y_test = y_all[test_mask].to_numpy()
        labels = sorted(y_all.dropna().unique().tolist())

        raw_cols = ranked[:k]
        model = joblib.load(ctx.run_dir / "checkpoints" / "lightgbm_dynamic470.joblib")
        X_test = df.loc[test_mask, raw_cols].to_numpy()
        preds = model.predict(X_test)
        report = full_classification_report(y_test, preds, labels)

        results = {"lightgbm": report}
        log.info(f"[lightgbm] TEST macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f} "
                 f"balanced_accuracy={report['balanced_accuracy']:.4f}")

        out_path = ctx.run_dir / "metrics" / "stage12_internal_evaluation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return {"internal_evaluation": str(out_path), "models_evaluated": ["lightgbm"]}
