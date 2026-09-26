"""Stage 04 — Deterministic, Leakage-Safe Train/Validation/Test Split.

Operates on MalDroid2020 only (AndMal2020 is never split — it is used whole, once, as
external validation, per the golden leakage rule). Known exact-duplicate rows (found in
the 470-feature dynamic file, the primary modality) are grouped so that both copies of a
duplicate always land in the same split — otherwise the same sample could appear in both
train and test, which would silently inflate the reported test accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage04_split")


def _duplicate_groups(dyn_470_path) -> dict[int, int]:
    """Return {row_index: group_id} where rows sharing an identical feature vector
    (in the primary 470-feature modality) share a group_id."""
    df = pd.read_csv(dyn_470_path)
    feat_cols = [c for c in df.columns if c != "Class"]
    keyed = df[feat_cols].apply(lambda r: hash(tuple(r.values.tolist())), axis=1)
    group_map: dict[Any, int] = {}
    row_to_group: dict[int, int] = {}
    next_gid = 0
    for i, k in enumerate(keyed):
        if k not in group_map:
            group_map[k] = next_gid
            next_gid += 1
        row_to_group[i] = group_map[k]
    return row_to_group


class SplitStage(Stage):
    name = "stage04_split"
    depends_on = ["stage02_manifest"]

    def config_section(self, cfg: dict) -> dict:
        return {"split": cfg["split"], "seed": cfg["seed"], "paths": {"maldroid_dynamic_470": cfg["paths"]["maldroid_dynamic_470"]}}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        seed = cfg["seed"]
        rng = np.random.RandomState(seed)

        manifest_path = ctx.run_dir / "manifests" / "maldroid_manifest.parquet"
        manifest = pd.read_parquet(manifest_path)

        log.info("Computing exact-duplicate groups from the primary (470-feature) dynamic file...")
        row_to_group = _duplicate_groups(resolve(cfg["paths"]["maldroid_dynamic_470"]))
        manifest["dup_group"] = manifest["sample_id"].map(row_to_group)
        n_groups = manifest["dup_group"].nunique()
        n_dupe_rows = len(manifest) - n_groups
        log.info(f"{n_dupe_rows} of {len(manifest)} rows are exact duplicates of another row "
                 f"({n_groups} distinct groups) — grouped so they cannot split across train/val/test.")

        train_frac, val_frac, test_frac = cfg["split"]["train_frac"], cfg["split"]["val_frac"], cfg["split"]["test_frac"]
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "split fractions must sum to 1.0"

        split_assignment = pd.Series(index=manifest["sample_id"], dtype=object)

        for label, sub in manifest.groupby("label_canonical"):
            groups_in_class = sub.groupby("dup_group")["sample_id"].apply(list).to_dict()
            group_ids = list(groups_in_class.keys())
            rng.shuffle(group_ids)

            n_total = sum(len(v) for v in groups_in_class.values())
            n_train_target = round(n_total * train_frac)
            n_val_target = round(n_total * val_frac)

            assigned = 0
            for gid in group_ids:
                members = groups_in_class[gid]
                if assigned < n_train_target:
                    split_name = "train"
                elif assigned < n_train_target + n_val_target:
                    split_name = "val"
                else:
                    split_name = "test"
                for sid in members:
                    split_assignment.loc[sid] = split_name
                assigned += len(members)

            log.info(f"  class '{label}': {n_total} samples -> "
                     f"train={sum(split_assignment.loc[manifest.loc[manifest['label_canonical']==label,'sample_id']]=='train')}, "
                     f"val={sum(split_assignment.loc[manifest.loc[manifest['label_canonical']==label,'sample_id']]=='val')}, "
                     f"test={sum(split_assignment.loc[manifest.loc[manifest['label_canonical']==label,'sample_id']]=='test')}")

        # ── Cross-class duplicate reconciliation ──────────────────────────────────
        # A dup_group may contain rows from different classes (same 470-feature
        # vector, different label).  The per-class loop above assigned them
        # independently, so they may land in different splits.  Force every
        # member of such a group to the majority split (tie → "train").
        manifest["_split_tmp"] = manifest["sample_id"].map(split_assignment)
        cross_class_groups = (
            manifest.groupby("dup_group")["label_canonical"].nunique()
        )
        cross_class_gids = cross_class_groups[cross_class_groups > 1].index.tolist()
        if cross_class_gids:
            log.warning(
                f"{len(cross_class_gids)} dup_group(s) span multiple classes "
                f"(same 470-feature vector, different label) — reconciling to majority split."
            )
            for gid in cross_class_gids:
                group_rows = manifest[manifest["dup_group"] == gid]
                current_splits = group_rows["_split_tmp"].value_counts()
                majority_split = current_splits.idxmax()  # ties broken by first-max
                member_sids = group_rows["sample_id"].tolist()
                classes_in_group = group_rows["label_canonical"].unique().tolist()
                log.warning(
                    f"  dup_group={gid}: classes={classes_in_group}, "
                    f"split votes={current_splits.to_dict()} → reassigning all to '{majority_split}'"
                )
                for sid in member_sids:
                    split_assignment.loc[sid] = majority_split
            manifest["_split_tmp"] = manifest["sample_id"].map(split_assignment)
        manifest.drop(columns=["_split_tmp"], inplace=True)

        manifest["split"] = manifest["sample_id"].map(split_assignment)
        assert manifest["split"].isna().sum() == 0, "every sample must receive a split assignment"

        out_path = ctx.run_dir / "manifests" / "maldroid_split_manifest.parquet"
        manifest[["sample_id", "label_canonical", "dup_group", "split"]].to_parquet(out_path, index=False)

        counts = manifest["split"].value_counts().to_dict()
        log.info(f"Split sizes: {counts}")

        # sanity: no dup_group spans more than one split
        span_check = manifest.groupby("dup_group")["split"].nunique()
        violations = int((span_check > 1).sum())
        if violations > 0:
            raise RuntimeError(f"LEAKAGE DETECTED: {violations} duplicate-groups span more than one split. Hard stop.")
        log.info("Leakage check passed: no duplicate group spans more than one split.")

        return {
            "split_manifest": str(out_path),
            "split_counts": {str(k): int(v) for k, v in counts.items()},
            "seed": seed,
            "duplicate_rows_grouped": int(n_dupe_rows),
        }
