"""Configuration loading and per-stage config hashing (used to detect stale artifacts)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from pipeline.core.paths import project_root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pipeline YAML config. Defaults to config/default.yaml."""
    cfg_path = Path(path) if path else project_root() / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(section: dict[str, Any]) -> str:
    """Deterministic short hash of a config section, used to detect when a stage's
    inputs have changed and its cached artifact is therefore stale."""
    blob = json.dumps(section, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
