"""KronoDroid Stage 07 — imbalance analysis (binary).

Binary label is near-balanced (measured in Doc 02 Part C.2: ~1.13x malware:benign on
real_device) — class_weight='balanced' at training time is sufficient; no synthetic
oversampling (SMOTE) is applied, unlike MalDroid's stage09 which faced a 3.1x imbalance.
This stage measures the TRAIN-split ratio directly rather than reusing the whole-dataset
ratio, and records the decision for the report.

Family-level imbalance (the severe long tail this project treats as its own contribution,
per the task brief) is handled in the family-track manifest (Stage 02's `family_label`
thresholding), not here — this stage is binary-only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.core.stage import Stage, StageContext


class KronoDroidImbalanceStage(Stage):
    name = "kd07_imbalance"
    depends_on = ["kd05_preprocess"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"imbalance": cfg["imbalance"]["binary"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        df = pd.read_parquet(ctx.run_dir / "artifacts" / "kd05_features_scaled.parquet")
        train = df[df["split"] == "train"]
        counts = train["label_binary"].value_counts()
        n_malware = int(counts.get(1, 0))
        n_benign = int(counts.get(0, 0))
        ratio = max(n_malware, n_benign) / min(n_malware, n_benign)

        strategy = ctx.cfg["imbalance"]["binary"]["strategy"]

        return {
            "train_n_malware": n_malware,
            "train_n_benign": n_benign,
            "train_imbalance_ratio": round(ratio, 4),
            "strategy_applied": strategy,
        }
