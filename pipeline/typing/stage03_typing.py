"""Stage 03 — Feature Typing.

MalDroid2020's static feature file mixes true numeric/binary columns with genuine text
columns (e.g. package-name-style strings) — this was discovered directly during dataset
forensics (Doc 02) and caused a real crash when everything was naively cast to numeric.
This stage classifies every static column BEFORE any numeric algorithm ever sees it, over
the FULL 11,598 rows (not a sample — the file is small enough to do this exhaustively),
processed in column batches so no more than a few thousand columns are in memory at once.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage03_typing")

BATCH_COLS = 3000


def _classify_column(series: pd.Series) -> str:
    # the repaired static parquet stores absent/blank flags as literal '' (empty string),
    # not NaN (Doc 02 A.9 repair preserved raw values exactly via keep_default_na=False) —
    # without this, every sparse binary column's blank cells fail numeric parsing and the
    # whole column gets misclassified as text. This was caught by inspecting Stage 01's
    # audit output, which showed ~99.8% of static columns as "text" — implausible given
    # direct earlier inspection showed most columns are binary 0/1 flags.
    series = series.replace("", None)
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unusable_all_missing"
    coerced = pd.to_numeric(non_null, errors="coerce")
    numeric_frac = coerced.notna().mean()
    if numeric_frac < 0.99:
        n_unique = non_null.nunique()
        if n_unique > 0.5 * len(non_null):
            return "high_cardinality_string"
        return "categorical_string"
    # numeric-ish. IMPORTANT: blank ('') means "feature absent" (a real 0), not "unknown" —
    # it must be FILLED with 0 here, not dropped, before checking variance/uniqueness.
    # Dropping it (as an earlier version of this function did) removes every "absent" row
    # from consideration, leaving only "present" (1.0) values behind — which makes every
    # genuinely binary flag column look artificially zero-variance. Caught via smoke test:
    # a synthetic binary column came back "zero_variance_numeric" for exactly this reason.
    full_numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if full_numeric.std() == 0:
        return "zero_variance_numeric"
    uniq = set(full_numeric.unique().tolist())
    if uniq.issubset({0, 1, 0.0, 1.0}):
        return "binary_numeric"
    return "count_or_continuous_numeric"


class FeatureTypingStage(Stage):
    name = "stage03_typing"
    depends_on = ["stage02_manifest"]

    def config_section(self, cfg: dict) -> dict:
        return {"paths": {"maldroid_static_repaired": cfg["paths"]["maldroid_static_repaired"]}}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        static_path = resolve(cfg["paths"]["maldroid_static_repaired"])
        pf = pq.ParquetFile(str(static_path))
        all_cols = pf.schema_arrow.names
        feat_cols = [c for c in all_cols if c != "Class"]
        idx_col = feat_cols[0]  # first column is the row-index/id column, not a real feature
        feat_cols = feat_cols[1:]

        typing: dict[str, str] = {idx_col: "identity_row_index"}
        counts: dict[str, int] = {}

        n_batches = (len(feat_cols) + BATCH_COLS - 1) // BATCH_COLS
        for bi, start in enumerate(range(0, len(feat_cols), BATCH_COLS)):
            batch = feat_cols[start:start + BATCH_COLS]
            tbl = pf.read(columns=batch)
            pdf = tbl.to_pandas()
            for c in pdf.columns:
                t = _classify_column(pdf[c])
                typing[c] = t
                counts[t] = counts.get(t, 0) + 1
            log.info(f"  typed batch {bi+1}/{n_batches} ({len(batch)} columns)")

        counts["identity_row_index"] = 1
        log.info(f"Feature typing summary: {counts}")

        usable_numeric = [c for c, t in typing.items() if t in ("binary_numeric", "count_or_continuous_numeric")]
        dropped = [c for c, t in typing.items() if t in ("categorical_string", "high_cardinality_string", "zero_variance_numeric", "unusable_all_missing")]

        out_path = ctx.run_dir / "artifacts" / "stage03_static_feature_types.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "type_counts": counts,
                "usable_numeric_columns": len(usable_numeric),
                "dropped_columns": len(dropped),
                "column_types": typing,
            }, f, indent=2)

        usable_list_path = ctx.run_dir / "artifacts" / "stage03_static_usable_numeric_columns.json"
        with open(usable_list_path, "w", encoding="utf-8") as f:
            json.dump(usable_numeric, f)

        log.info(f"{len(usable_numeric)} of {len(feat_cols)} static columns are usable numeric features; "
                  f"{len(dropped)} dropped (text/high-cardinality/zero-variance/all-missing) — decision logged, nothing silently coerced.")

        return {
            "typing_report": str(out_path),
            "usable_numeric_columns_list": str(usable_list_path),
            "type_counts": counts,
            "usable_numeric_count": len(usable_numeric),
            "dropped_count": len(dropped),
        }
