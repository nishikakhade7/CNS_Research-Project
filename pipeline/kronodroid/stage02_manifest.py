"""KronoDroid Stage 02 — canonical manifest: one row per sample, identity + label columns.

Mirrors the role of pipeline/manifest/stage02_manifest.py for MalDroid (identity table that
every later stage joins against), rebuilt for KronoDroid's own schema since real `sha256`
exists here (MalDroid's row-order content-hash workaround does not apply).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.core.stage import Stage, StageContext

METADATA_COLS = [
    "Package", "sha256", "CFileSize", "UFileSize", "EarliestModDate", "HighestModDate",
    "Scanners", "Detection_Ratio", "MalFamily",
]


class KronoDroidManifestStage(Stage):
    name = "kd02_manifest"
    depends_on = ["kd01_audit"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"family_min_samples": cfg["label_map"]["family_min_samples"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        audit_record = ctx.cfg.get("_kd01_artifacts")  # not used; path is deterministic
        audited_path = ctx.run_dir / "artifacts" / "kd01_audited_real_device.parquet"
        df = pd.read_parquet(audited_path)

        manifest = pd.DataFrame({
            "sample_id": df["sha256"],
            "source_dataset": "kronodroid_real_device",
            "Package": df["Package"],
            "label_binary_raw": df["Malware"].astype(int),
            "label_binary_canonical": df["Malware"].map({0: "Benign", 1: "Malware"}),
            "MalFamily_raw": df["MalFamily"],
        })

        # Family-level label: NaN family (malware with unrecorded family, or legitimate
        # apps by definition) is kept as an explicit "Benign_or_Unlabeled" bucket, never
        # silently coerced into a real family name.
        min_n = ctx.cfg["label_map"]["family_min_samples"]
        family_counts = manifest.loc[manifest["label_binary_raw"] == 1, "MalFamily_raw"].value_counts()
        keep_families = set(family_counts[family_counts >= min_n].index)

        def _family_label(row: pd.Series) -> str:
            if row["label_binary_raw"] == 0:
                return "Benign"
            fam = row["MalFamily_raw"]
            if pd.isna(fam):
                return "Other"
            return fam if fam in keep_families else "Other"

        manifest["family_label"] = manifest.apply(_family_label, axis=1)

        out_path = ctx.run_dir / "manifests" / "kd02_manifest.parquet"
        manifest.to_parquet(out_path, index=False)

        family_dist = manifest["family_label"].value_counts().to_dict()

        return {
            "n_samples": len(manifest),
            "n_malware": int((manifest["label_binary_raw"] == 1).sum()),
            "n_benign": int((manifest["label_binary_raw"] == 0).sum()),
            "family_min_samples_threshold": min_n,
            "n_family_classes_kept": len(keep_families),
            "n_family_classes_raw": int(manifest.loc[manifest["label_binary_raw"] == 1, "MalFamily_raw"].nunique()),
            "family_label_distribution_top10": dict(sorted(family_dist.items(), key=lambda kv: -kv[1])[:10]),
            "manifest_parquet": str(out_path),
        }
