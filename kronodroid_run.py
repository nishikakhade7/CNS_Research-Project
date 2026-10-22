"""Single entry point for the KronoDroid pipeline track — a SEPARATE, PARALLEL pipeline to
one_run.py's MalDroid2020/AndMal2020 pipeline. Does not import or modify anything in that
pipeline's registry; shares only pipeline/core (dataset-agnostic Stage/config/paths/logging
infrastructure).

Usage:
    python kronodroid_run.py --all
    python kronodroid_run.py --stage kd03_typing
    python kronodroid_run.py --list
"""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.core.config import load_config
from pipeline.core.environment import capture_environment
from pipeline.core.paths import project_root, run_dir
from pipeline.core.runlog import get_stage_logger, setup_logging
from pipeline.core.stage import Stage, StageContext, StageDependencyError, StageExecutionError

from pipeline.kronodroid.stage01_audit import KronoDroidAuditStage
from pipeline.kronodroid.stage02_manifest import KronoDroidManifestStage
from pipeline.kronodroid.stage03_typing import KronoDroidTypingStage
from pipeline.kronodroid.stage04_split import KronoDroidSplitStage
from pipeline.kronodroid.stage05_preprocess import KronoDroidPreprocessStage
from pipeline.kronodroid.stage06_selection import KronoDroidSelectionStage
from pipeline.kronodroid.stage07_imbalance import KronoDroidImbalanceStage
from pipeline.kronodroid.stage08_train import KronoDroidTrainStage
from pipeline.kronodroid.stage09_eval import KronoDroidEvalStage

STAGE_REGISTRY: dict[str, type[Stage]] = {
    "kd01_audit": KronoDroidAuditStage,
    "kd02_manifest": KronoDroidManifestStage,
    "kd03_typing": KronoDroidTypingStage,
    "kd04_split": KronoDroidSplitStage,
    "kd05_preprocess": KronoDroidPreprocessStage,
    "kd06_selection": KronoDroidSelectionStage,
    "kd07_imbalance": KronoDroidImbalanceStage,
    "kd08_train": KronoDroidTrainStage,
    "kd09_eval": KronoDroidEvalStage,
}

ORDER = list(STAGE_REGISTRY.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="KronoDroid pipeline orchestrator (Group 3)")
    parser.add_argument("--all", action="store_true", help="Run every stage in order")
    parser.add_argument("--stage", type=str, default=None, help="Run a single stage")
    parser.add_argument("--force-stage", action="store_true", help="Force re-run of the target stage(s)")
    parser.add_argument("--config", type=str, default=str(project_root() / "config" / "kronodroid.yaml"))
    parser.add_argument("--run-name", type=str, default="kronodroid")
    parser.add_argument("--list", action="store_true")
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
        log.error(f"Unknown stage '{args.stage}'. Use --list.")
        return 2

    log.info(f"Run directory: {rdir}")
    log.info(f"Targets: {targets}")

    for stage_name in targets:
        stage = STAGE_REGISTRY[stage_name]()
        try:
            stage.execute(ctx)
        except (StageDependencyError, StageExecutionError) as e:
            log.error(f"HARD STOP at '{stage_name}': {e}")
            return 3

    log.info("KronoDroid run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
