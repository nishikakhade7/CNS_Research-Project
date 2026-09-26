"""Single entry point for the Group 3 Android-malware pipeline.

Usage:
    python one_run.py --all
    python one_run.py --stage stage01_audit
    python one_run.py --stage stage03_typing --force-stage
    python one_run.py --list

Every stage is resumable: if it already SUCCEEDED with the same config, it is skipped.
If a prerequisite stage did not succeed, the run hard-stops with a clear error instead of
silently continuing or re-deriving the missing artifact.
"""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.core.config import load_config
from pipeline.core.environment import capture_environment
from pipeline.core.paths import run_dir
from pipeline.core.runlog import get_stage_logger, setup_logging
from pipeline.core.stage import Stage, StageContext, StageDependencyError, StageExecutionError

from pipeline.audit.stage01_audit import DataAuditStage
from pipeline.manifest.stage02_manifest import CanonicalManifestStage
from pipeline.typing.stage03_typing import FeatureTypingStage
from pipeline.splitting.stage04_split import SplitStage
from pipeline.static.stage05_static import StaticPreprocessingStage
from pipeline.dynamic.stage06_dynamic import DynamicPreprocessingStage
from pipeline.fusion.stage07_fusion import FusionStage
from pipeline.selection.stage08_selection import FeatureSelectionStage
from pipeline.imbalance.stage09_imbalance import ImbalanceStage
from pipeline.training.stage11_train import TrainingStage
from pipeline.evaluation.stage12_eval import InternalEvaluationStage
from pipeline.evaluation.stage13_ablation import AblationStage
from pipeline.evaluation.stage14_lock import LockStage
from pipeline.external.stage15_external import ExternalValidationStage
from pipeline.reporting.stage16_report import FinalReportStage

STAGE_REGISTRY: dict[str, type[Stage]] = {
    "stage01_audit": DataAuditStage,
    "stage02_manifest": CanonicalManifestStage,
    "stage03_typing": FeatureTypingStage,
    "stage04_split": SplitStage,
    "stage05_static": StaticPreprocessingStage,
    "stage06_dynamic": DynamicPreprocessingStage,
    "stage07_fusion": FusionStage,
    "stage08_selection": FeatureSelectionStage,
    "stage09_imbalance": ImbalanceStage,
    "stage11_train": TrainingStage,
    "stage12_eval": InternalEvaluationStage,
    "stage13_ablation": AblationStage,
    "stage14_lock": LockStage,
    "stage15_external": ExternalValidationStage,
    "stage16_report": FinalReportStage,
}

ORDER = list(STAGE_REGISTRY.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="Group 3 Android malware pipeline orchestrator")
    parser.add_argument("--all", action="store_true", help="Run every stage in order")
    parser.add_argument("--stage", type=str, default=None, help="Run a single stage (and its unmet dependencies are NOT auto-run — they must already have succeeded)")
    parser.add_argument("--force-stage", action="store_true", help="Force re-run of the target stage(s) even if already SUCCESS")
    parser.add_argument("--config", type=str, default=None, help="Path to an alternate config YAML")
    parser.add_argument("--run-name", type=str, default="latest", help="Run directory name under runs/ (default: latest)")
    parser.add_argument("--list", action="store_true", help="List all registered stages and exit")
    args = parser.parse_args()

    if args.list:
        for s in ORDER:
            print(s)
        return 0

    cfg = load_config(args.config)
    rdir = run_dir(args.run_name)
    setup_logging(rdir)
    log = get_stage_logger("orchestrator")

    with open(rdir / "config.yaml.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    with open(rdir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(capture_environment(), f, indent=2, default=str)

    force_set = set()
    if args.stage and args.force_stage:
        force_set.add(args.stage)
    if args.all and args.force_stage:
        force_set = set(ORDER)

    ctx = StageContext(cfg=cfg, run_dir=rdir, force_stages=force_set)

    if not args.all and not args.stage:
        parser.print_help()
        return 1

    targets = ORDER if args.all else [args.stage]
    if args.stage and args.stage not in STAGE_REGISTRY:
        log.error(f"Unknown stage '{args.stage}'. Use --list to see valid stage names.")
        return 2

    log.info(f"Run directory: {rdir}")
    log.info(f"Targets: {targets}")

    for stage_name in targets:
        stage_cls = STAGE_REGISTRY[stage_name]
        stage = stage_cls()
        try:
            stage.execute(ctx)
        except (StageDependencyError, StageExecutionError) as e:
            log.error(f"HARD STOP at '{stage_name}': {e}")
            return 3

    log.info("Run complete — all requested stages finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
