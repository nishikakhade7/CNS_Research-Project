"""Stage 07 — Static + Dynamic Fusion (MalDroid2020 only).

This is only possible because MalDroid2020's static and dynamic files describe the same
11,598 apps in the same row order (Doc 02 A.7) — an assumption validated empirically
(92.8% vs 33.7% baseline classifier, Stage-independent finding from prior forensic work),
NOT a cryptographic/hash-based join, because the static file carries no sample hash.
AndMal2020 is never fused (its static and dynamic files are different, non-overlapping
sample populations — Doc 02 B.8) — this stage only ever touches MalDroid data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage07_fusion")


class FusionStage(Stage):
    name = "stage07_fusion"
    depends_on = ["stage05_static", "stage06_dynamic"]

    def config_section(self, cfg: dict) -> dict:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        static_df = pd.read_parquet(ctx.run_dir / "artifacts" / "stage05_static_features.parquet")
        dyn_df = pd.read_parquet(ctx.run_dir / "artifacts" / "stage06_dynamic_470_features.parquet")

        if len(static_df) != len(dyn_df):
            raise RuntimeError(
                f"LEAKAGE/ALIGNMENT RISK: static branch has {len(static_df)} rows, "
                f"dynamic-470 branch has {len(dyn_df)} rows — row-order fusion requires equal counts. Hard stop."
            )

        static_cols = [c for c in static_df.columns if c not in ("sample_id", "split")]
        dyn_cols = [c for c in dyn_df.columns if c not in ("sample_id", "split", "Class")]

        static_renamed = static_df[["sample_id", "split"] + static_cols].rename(
            columns={c: f"static__{c}" for c in static_cols}
        )
        dyn_renamed = dyn_df[["sample_id", "Class"] + dyn_cols].rename(
            columns={c: f"dyn470__{c}" for c in dyn_cols}
        )

        fused = static_renamed.merge(dyn_renamed, on="sample_id", how="inner")
        if len(fused) != len(static_df):
            raise RuntimeError(
                f"LEAKAGE/ALIGNMENT RISK: fused row count ({len(fused)}) does not match "
                f"static row count ({len(static_df)}) after merge on sample_id — row-order "
                f"alignment assumption may be violated. Hard stop."
            )

        out_path = ctx.run_dir / "artifacts" / "stage07_fused_features.parquet"
        fused.to_parquet(out_path, index=False)
        log.info(f"Fused static+dynamic matrix: {fused.shape} -> {out_path} "
                 f"({len(static_cols)} static cols + {len(dyn_cols)} dynamic-470 cols, row-order-aligned)")

        return {"fused_features": str(out_path), "rows": int(len(fused)),
                "static_columns": len(static_cols), "dynamic_columns": len(dyn_cols)}
