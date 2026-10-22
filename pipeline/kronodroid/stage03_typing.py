"""KronoDroid Stage 03 — static/dynamic/metadata column manifest, built by NAME-MATCHING
every column against the reference lists in reference_data.py, not by column position.

Any column matching none of the reference sets is reported in `unclassified_columns` and
excluded from both the static and dynamic feature groups — never silently guessed into
one bucket, per this project's no-fabrication rule.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from pipeline.core.stage import Stage, StageContext
from pipeline.kronodroid.reference_data import (
    APK_STRUCTURAL_COLS,
    DYNAMIC_BEHAVIORAL_COLS,
    KNOWN_ANDROID_PERMISSIONS,
    KNOWN_SYSCALL_NAMES,
    METADATA_COLS,
    PERMISSION_SUMMARY_COLS,
    SYSCALL_RESERVED_OR_SUMMARY,
)


class KronoDroidTypingStage(Stage):
    name = "kd03_typing"
    depends_on = ["kd01_audit"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        audited_path = ctx.run_dir / "artifacts" / "kd01_audited_real_device.parquet"
        df = pd.read_parquet(audited_path)
        all_cols = list(df.columns)

        dynamic_syscall = [c for c in all_cols if c in KNOWN_SYSCALL_NAMES]
        dynamic_reserved = [c for c in all_cols if c in SYSCALL_RESERVED_OR_SUMMARY]
        dynamic_behavioral = [c for c in all_cols if c in DYNAMIC_BEHAVIORAL_COLS]
        static_permission = [c for c in all_cols if c in KNOWN_ANDROID_PERMISSIONS]
        static_permission_summary = [c for c in all_cols if c in PERMISSION_SUMMARY_COLS]
        static_apk_structural = [c for c in all_cols if c in APK_STRUCTURAL_COLS]
        metadata = [c for c in all_cols if c in METADATA_COLS]

        classified = (
            set(dynamic_syscall) | set(dynamic_reserved) | set(dynamic_behavioral)
            | set(static_permission) | set(static_permission_summary) | set(static_apk_structural)
            | set(metadata)
        )
        unclassified = [c for c in all_cols if c not in classified]

        dynamic_cols = dynamic_syscall + dynamic_reserved + dynamic_behavioral
        static_cols = static_permission + static_permission_summary + static_apk_structural

        # Cross-check for name collisions between reference lists (would indicate a bug
        # in reference_data.py, not a property of the data) — hard stop, not a warning.
        overlap = (set(dynamic_cols) & set(static_cols)) | (set(dynamic_cols) & set(metadata)) | (set(static_cols) & set(metadata))
        if overlap:
            raise ValueError(f"Column(s) matched more than one reference group — fix reference_data.py: {overlap}")

        manifest = {
            "dynamic_syscall_known_name": sorted(dynamic_syscall),
            "dynamic_syscall_reserved_or_summary": sorted(dynamic_reserved),
            "dynamic_behavioral": sorted(dynamic_behavioral),
            "static_permission": sorted(static_permission),
            "static_permission_summary": sorted(static_permission_summary),
            "static_apk_structural": sorted(static_apk_structural),
            "metadata": sorted(metadata),
            "unclassified": sorted(unclassified),
        }
        manifest_path = ctx.run_dir / "manifests" / "kd03_column_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        counts = {k: len(v) for k, v in manifest.items()}
        counts["total_columns"] = len(all_cols)
        counts["total_dynamic"] = len(dynamic_cols)
        counts["total_static"] = len(static_cols)
        # Paper's claimed split (Guerra-Manzanares et al. 2021): 200 static + 289 dynamic.
        counts["paper_claimed_static"] = 200
        counts["paper_claimed_dynamic"] = 289

        if unclassified:
            raise ValueError(
                f"{len(unclassified)} column(s) matched no reference list: {unclassified}. "
                "Manual review required before proceeding — see manifest file for the full picture."
            )

        return {"column_manifest": str(manifest_path), **counts}
