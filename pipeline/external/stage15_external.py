"""Stage 15 — AndMal2020 External Validation.

Loads ONLY the frozen Stage 14 model artifact. Never refits anything on AndMal data.

CRITICAL, VERIFIED CONSTRAINT (checked here programmatically, not assumed):
A frozen model requires its EXACT trained feature columns, by name, to be present in the
external dataset — "ignore unsupported extra AndMal features" only works if the model's
OWN features are a subset of what AndMal provides. This stage computes that name
intersection for real before attempting any prediction. If the locked modality is
`static`, alignment is impossible by construction: AndMal2020's static files are
headerless (Doc 02 B.3) and carry no feature names on the AndMal side to match against,
so there is nothing for MalDroid's static column names to align to. If the locked modality
is a dynamic view, MalDroid's dynamic vocabulary (syscall/binder names) and AndMal's
dynamic vocabulary (Memory/API/Battery/Logcat/Process names) are measured by entirely
different instrumentation and are checked for name overlap rather than assumed disjoint.

If the required feature columns are not available in AndMal by name, this stage does NOT
fabricate a workaround (zero-filling missing columns would silently change what the model
is actually being asked to classify). It reports the exact reason external validation could
not proceed for this locked configuration — a documented negative finding, not a fabricated
number.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext
from pipeline.evaluation.metrics import full_classification_report

log = get_stage_logger("stage15_external")


def _andmal_dynamic_columns(cfg: dict) -> list[str]:
    d = resolve(cfg["paths"]["andmal_dynamic"])
    any_file = next(d.glob("*.csv"))
    return list(pd.read_csv(any_file, nrows=1).columns)


def _load_andmal_dynamic_mapped(cfg: dict, common_classes: list[str]) -> pd.DataFrame:
    d = resolve(cfg["paths"]["andmal_dynamic"])
    andmal_map = cfg["label_map"]["andmal_to_maldroid"]
    frames = []
    for csv_path in sorted(d.glob("*.csv")):
        # filenames like Adware_before_reboot_Cat.csv / Trojan_Banker_after_reboot_Cat.csv
        stem = csv_path.stem.replace("_before_reboot_Cat", "").replace("_after_reboot_Cat", "")
        stem_norm = stem.replace("Trojan_Banker", "Banker").replace("Trojan_SMS", "SMS").replace("_", "")
        canonical = None
        for raw_key, canon in andmal_map.items():
            if raw_key.replace("_", "").lower() == stem_norm.lower():
                canonical = canon
                break
        if canonical is None or canonical not in common_classes:
            continue
        df = pd.read_csv(csv_path)
        df["label_canonical"] = canonical
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


class ExternalValidationStage(Stage):
    name = "stage15_external"
    depends_on = ["stage14_lock"]

    def config_section(self, cfg: dict) -> dict:
        return {"andmal_external": cfg["andmal_external"]}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.cfg
        with open(ctx.run_dir / "artifacts" / "stage14_lock_manifest.json", encoding="utf-8") as f:
            lock = json.load(f)

        modality = lock["selected_modality"]
        required_cols = lock["feature_columns"]
        # strip the "__scaled" / "static__" / "dyn470__" prefixes/suffixes we added ourselves,
        # to get back to the real underlying feature name for cross-dataset name matching
        def base_name(c: str) -> str:
            c = c.replace("__scaled", "")
            c = c.replace("static__", "").replace("dyn470__", "")
            return c
        required_base = [base_name(c) for c in required_cols]

        log.info(f"Locked model: {lock['selected_model']} / modality={modality}, "
                 f"{len(required_base)} required feature columns.")

        if modality == "static":
            report = {
                "external_validation_possible": False,
                "reason": (
                    "Locked modality is 'static'. AndMal2020's static files are headerless "
                    "(Doc 02 B.3) — column position 0..9502, no feature NAMES exist on the "
                    "AndMal side. MalDroid's static feature names (even after repair) cannot "
                    "be matched to unnamed AndMal columns without an external, undownloaded "
                    "AndMal feature-name manifest. This is a genuine, structural limitation of "
                    "the two dataset releases, not a bug in this pipeline."
                ),
            }
            log.warning(report["reason"])
        else:
            andmal_cols = _andmal_dynamic_columns(cfg)
            overlap = [c for c in required_base if c in andmal_cols]
            log.info(f"Name-overlap check: {len(overlap)} of {len(required_base)} required "
                     f"MalDroid dynamic feature names found (by literal name) in AndMal2020's "
                     f"dynamic column set ({len(andmal_cols)} columns available).")

            MIN_OVERLAP_FRACTION = 1.0  # the frozen model's exact input shape must be reproducible
            if len(overlap) < len(required_base) * MIN_OVERLAP_FRACTION:
                report = {
                    "external_validation_possible": False,
                    "required_features": len(required_base),
                    "matched_by_name_in_andmal": len(overlap),
                    "andmal_dynamic_columns_available": len(andmal_cols),
                    "reason": (
                        "The frozen model requires its exact trained feature columns to be present "
                        "in AndMal2020 by name. MalDroid's dynamic features are syscall/binder names "
                        "(e.g. process/IPC-level events); AndMal2020's dynamic features are memory/API/"
                        "battery/logcat/process-count names — different instrumentation entirely. "
                        "Zero-filling the missing columns was considered and rejected: it would silently "
                        "change what the model is being asked to classify, not perform a real evaluation. "
                        "This is reported as a genuine cross-dataset feature-space incompatibility, not "
                        "fabricated as a passing external-validation score."
                    ),
                }
                log.warning(report["reason"])
            else:
                andmal_df = _load_andmal_dynamic_mapped(cfg, cfg["andmal_external"]["common_classes"])
                bundle = joblib.load(resolve(lock["frozen_artifact"]))
                model = bundle["model"]
                X_ext = andmal_df[required_base].to_numpy()
                y_ext = andmal_df["label_canonical"].to_numpy()
                preds = model.predict(X_ext)
                labels = sorted(set(y_ext.tolist()))
                metrics = full_classification_report(y_ext, preds, labels)
                report = {"external_validation_possible": True, "metrics": metrics, "n_samples": int(len(andmal_df))}
                log.info(f"External macro_f1={metrics['macro_f1']:.4f} on {len(andmal_df)} AndMal samples.")

        out_path = ctx.run_dir / "metrics" / "stage15_external_validation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return {"external_validation_report": str(out_path), "possible": report["external_validation_possible"]}
