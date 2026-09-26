"""Central path resolution. No hardcoded absolute paths anywhere else in the codebase."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (three levels up from this file: pipeline/core/paths.py)."""
    return Path(__file__).resolve().parents[2]


def resolve(relative: str) -> Path:
    """Resolve a config-declared relative path against the project root."""
    return project_root() / relative


def run_dir(run_name: str = "latest") -> Path:
    """Return the active run directory, creating its standard subfolders if missing."""
    base = project_root() / "runs" / run_name
    for sub in ["logs", "artifacts", "checkpoints", "metrics", "plots", "diagrams", "manifests", "reports"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base
