"""status.json read/write. This file is the single source of truth for which stages have
completed, what their config hash was, and what artifacts they produced — it is what makes
`--stageNN` resumable and what lets a later stage hard-stop if a prerequisite is missing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

StageStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"]


def _status_path(run_directory: Path) -> Path:
    return run_directory / "status.json"


def load_status(run_directory: Path) -> dict[str, Any]:
    p = _status_path(run_directory)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_status(run_directory: Path, status: dict[str, Any]) -> None:
    p = _status_path(run_directory)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def get_stage_record(run_directory: Path, stage_name: str) -> dict[str, Any] | None:
    return load_status(run_directory).get(stage_name)


def update_stage_record(run_directory: Path, stage_name: str, record: dict[str, Any]) -> None:
    status = load_status(run_directory)
    status[stage_name] = record
    save_status(run_directory, status)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
