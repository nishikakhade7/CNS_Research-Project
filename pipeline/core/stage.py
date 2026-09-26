"""Stage base class: the resumability / hard-stop-on-failure / checkpoint engine.

Every pipeline stage subclasses `Stage`. The orchestrator (one_run.py) never runs a stage's
logic directly — it always goes through `Stage.execute()`, which is what enforces:
  - dependency checking (hard stop if a prerequisite stage did not succeed)
  - skip-if-already-succeeded-with-matching-config (unless --force-stage)
  - full traceback capture + status.json update on failure, then re-raise (hard stop)
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.core.config import config_hash
from pipeline.core.runlog import get_stage_logger
from pipeline.core.status import get_stage_record, now_iso, update_stage_record


class StageDependencyError(RuntimeError):
    """Raised when a stage's prerequisite has not succeeded — this is a hard stop."""


class StageExecutionError(RuntimeError):
    """Raised (after logging) when a stage's own logic fails — this is a hard stop."""


@dataclass
class StageContext:
    """Shared state passed to every stage: config, run directory, and force flags."""

    cfg: dict[str, Any]
    run_dir: Path
    force_stages: set[str] = field(default_factory=set)


class Stage:
    name: str = "unnamed_stage"
    depends_on: list[str] = []

    def config_section(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Return the slice of config relevant to this stage, for hashing. Override per stage."""
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        """Do the actual work. Return a dict of artifact metadata to persist in status.json."""
        raise NotImplementedError

    def validate_artifacts(self, artifacts: dict[str, Any]) -> bool:
        """Check that previously-recorded artifact paths still exist on disk."""
        for _, v in artifacts.items():
            if isinstance(v, str) and ("/" in v or "\\" in v) and v.endswith((".parquet", ".csv", ".joblib", ".pt", ".json")):
                if not Path(v).exists():
                    return False
        return True

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        log = get_stage_logger(self.name)
        force = self.name in ctx.force_stages

        for dep in self.depends_on:
            dep_record = get_stage_record(ctx.run_dir, dep)
            if dep_record is None or dep_record.get("status") != "SUCCESS":
                msg = (
                    f"HARD STOP: stage '{self.name}' requires prerequisite stage '{dep}' "
                    f"to have SUCCEEDED, but its status is "
                    f"{'MISSING' if dep_record is None else dep_record.get('status')}. "
                    f"Run: python one_run.py --stage {dep}"
                )
                log.error(msg)
                raise StageDependencyError(msg)

        this_cfg_hash = config_hash(self.config_section(ctx.cfg))
        existing = get_stage_record(ctx.run_dir, self.name)

        if not force and existing and existing.get("status") == "SUCCESS" and existing.get("config_hash") == this_cfg_hash:
            if self.validate_artifacts(existing.get("artifacts", {})):
                log.info(f"SKIP — stage '{self.name}' already SUCCEEDED with matching config, artifacts validated.")
                return existing
            else:
                log.warning(f"stage '{self.name}' recorded SUCCESS but an artifact is missing on disk — re-running.")

        log.info(f"START — stage '{self.name}'" + (" (forced re-run)" if force else ""))
        start = time.time()
        update_stage_record(
            ctx.run_dir, self.name,
            {"status": "RUNNING", "start_time": now_iso(), "config_hash": this_cfg_hash},
        )

        try:
            artifacts = self.run(ctx)
        except Exception as e:
            duration = time.time() - start
            tb = traceback.format_exc()
            log.error(f"FAILED — stage '{self.name}' after {duration:.1f}s: {e}")
            log.error(tb)
            update_stage_record(
                ctx.run_dir, self.name,
                {
                    "status": "FAILED",
                    "start_time": now_iso(),
                    "duration_sec": duration,
                    "config_hash": this_cfg_hash,
                    "error": str(e),
                    "traceback": tb,
                },
            )
            raise StageExecutionError(f"Stage '{self.name}' failed: {e}") from e

        duration = time.time() - start
        record = {
            "status": "SUCCESS",
            "start_time": now_iso(),
            "duration_sec": round(duration, 2),
            "config_hash": this_cfg_hash,
            "artifacts": artifacts,
        }
        update_stage_record(ctx.run_dir, self.name, record)
        log.info(f"DONE  — stage '{self.name}' in {duration:.1f}s")
        return record
