"""KronoDroid Stage 04 — leakage-safe train/val/test split.

Simpler than MalDroid's stage04 (which had to hash each row's feature vector because no
real sample identifier existed): KronoDroid has a genuine `sha256` per app, and Stage 01
already guaranteed zero duplicate sha256 in the cleaned real_device set, so each sample is
already its own group — "hash-grouped" here reduces to a stratified split keyed on sha256,
with an explicit post-split assertion that no sha256 appears in more than one split (the
same hard-stop discipline as MalDroid/AndMal, kept even though it's structurally
guaranteed here, in case a future upstream data change reintroduces duplicates).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.core.stage import Stage, StageContext


class KronoDroidSplitStage(Stage):
    name = "kd04_split"
    depends_on = ["kd02_manifest"]

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"split": cfg["split"], "seed": cfg["seed"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        manifest = pd.read_parquet(ctx.run_dir / "manifests" / "kd02_manifest.parquet")

        train_f = cfg["split"]["train_frac"]
        val_f = cfg["split"]["val_frac"]
        test_f = cfg["split"]["test_frac"]
        if abs(train_f + val_f + test_f - 1.0) > 1e-9:
            raise ValueError("train/val/test fractions must sum to 1.0")

        rng = np.random.default_rng(cfg["seed"])
        split_assignment = pd.Series(index=manifest.index, dtype="object")

        # Stratify by label_binary: shuffle within each class, assign by cumulative fraction.
        for label_val, group in manifest.groupby("label_binary_raw"):
            idx = group.index.to_numpy().copy()
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(round(n * train_f))
            n_val = int(round(n * val_f))
            split_assignment.loc[idx[:n_train]] = "train"
            split_assignment.loc[idx[n_train:n_train + n_val]] = "val"
            split_assignment.loc[idx[n_train + n_val:]] = "test"

        manifest["split"] = split_assignment

        # Hard-stop leakage assertion: every sha256 (= sample_id here) in exactly one split.
        dup_split = manifest.groupby("sample_id")["split"].nunique()
        leaking = dup_split[dup_split > 1]
        if len(leaking) > 0:
            raise ValueError(f"{len(leaking)} sample_id(s) span more than one split — leakage bug.")

        out_path = ctx.run_dir / "manifests" / "kd04_split_manifest.parquet"
        manifest.to_parquet(out_path, index=False)

        counts = manifest.groupby(["split", "label_binary_canonical"]).size().unstack(fill_value=0)
        split_label_counts = {str(s): {str(k): int(v) for k, v in row.items()} for s, row in counts.iterrows()}

        return {
            "split_manifest": str(out_path),
            "n_train": int((manifest["split"] == "train").sum()),
            "n_val": int((manifest["split"] == "val").sum()),
            "n_test": int((manifest["split"] == "test").sum()),
            "split_label_counts": split_label_counts,
        }
