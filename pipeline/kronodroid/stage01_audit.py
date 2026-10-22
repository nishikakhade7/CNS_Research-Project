"""KronoDroid Stage 01 — audit the real_device files against the CONFIRMED findings in
Docs/02_Deep_Dataset_Forensic_Findings.md Part C, and drop the 4 known label-conflict hashes.

This stage re-measures every claim rather than trusting the doc: if the raw files ever
change (updated release, re-download), this stage fails loudly instead of silently
propagating stale assumptions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.core.paths import resolve
from pipeline.core.stage import Stage, StageContext


class KronoDroidAuditStage(Stage):
    name = "kd01_audit"
    depends_on: list[str] = []

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "paths": {
                k: v for k, v in cfg["paths"].items() if k.startswith("kronodroid_real")
            },
            "label_conflict_hashes": sorted(cfg["label_conflict_hashes"]),
        }

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        mal_path = resolve(cfg["paths"]["kronodroid_real_malware"])
        leg_path = resolve(cfg["paths"]["kronodroid_real_legitimate"])

        mal = pd.read_csv(mal_path, low_memory=False)
        leg = pd.read_csv(leg_path, low_memory=False)

        # --- Re-confirm Doc 02 Part C.1 / C.2 row counts; hard stop on drift ---
        expected_mal_rows, expected_leg_rows = 41382, 36755
        if len(mal) != expected_mal_rows:
            raise ValueError(
                f"real_malware_v1.csv has {len(mal)} rows, expected {expected_mal_rows} "
                "(Doc 02 Part C.1) — raw file has changed; re-run the forensic audit before trusting this pipeline."
            )
        if len(leg) != expected_leg_rows:
            raise ValueError(
                f"real_legitimate_v1.csv has {len(leg)} rows, expected {expected_leg_rows} "
                "(Doc 02 Part C.1) — raw file has changed; re-run the forensic audit before trusting this pipeline."
            )
        if mal["Malware"].nunique() != 1 or mal["Malware"].iloc[0] != 1:
            raise ValueError("real_malware_v1.csv has a non-uniform or wrong Malware label.")
        if leg["Malware"].nunique() != 1 or leg["Malware"].iloc[0] != 0:
            raise ValueError("real_legitimate_v1.csv has a non-uniform or wrong Malware label.")

        # --- Re-confirm no WITHIN-file duplicate sha256 (Doc 02 Part C.1) ---
        for name, df in [("real_malware", mal), ("real_legitimate", leg)]:
            n_dup = int(df["sha256"].duplicated().sum())
            if n_dup != 0:
                raise ValueError(f"{name}: expected 0 duplicate sha256, found {n_dup} — re-audit before proceeding.")

        combined = pd.concat([mal, leg], ignore_index=True)
        raw_rows = len(combined)

        # --- Drop the 4 confirmed label-conflict hashes (Doc 02 Part C.5) ---
        conflict_hashes = set(cfg["label_conflict_hashes"])
        present_conflicts = conflict_hashes & set(combined["sha256"])
        if present_conflicts != conflict_hashes:
            missing = conflict_hashes - present_conflicts
            raise ValueError(
                f"Expected 4 known label-conflict hashes in the combined real_device file; "
                f"missing {missing} — raw file may have changed since the Doc 02 audit."
            )
        n_conflict_rows_dropped = int(combined["sha256"].isin(conflict_hashes).sum())
        combined = combined[~combined["sha256"].isin(conflict_hashes)].reset_index(drop=True)

        # Sanity: after dropping, no sha256 should repeat (each hash lived in exactly one file).
        n_dup_after = int(combined["sha256"].duplicated().sum())
        if n_dup_after != 0:
            raise ValueError(f"Unexpected {n_dup_after} duplicate sha256 remain after dropping label conflicts.")

        out_path = ctx.run_dir / "artifacts" / "kd01_audited_real_device.parquet"
        combined.to_parquet(out_path, index=False)

        return {
            "raw_malware_rows": expected_mal_rows,
            "raw_legitimate_rows": expected_leg_rows,
            "raw_combined_rows": raw_rows,
            "label_conflict_rows_dropped": n_conflict_rows_dropped,
            "clean_combined_rows": len(combined),
            "n_columns": combined.shape[1],
            "audited_parquet": str(out_path),
        }
