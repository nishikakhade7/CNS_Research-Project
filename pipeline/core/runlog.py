"""loguru-based run logging. No print() anywhere in the pipeline — this module is the only
place stdout/file sinks are configured."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_CONFIGURED_RUN: Path | None = None


def setup_logging(run_directory: Path) -> None:
    """Attach console + run.log + error.log sinks. Idempotent per run directory."""
    global _CONFIGURED_RUN
    if _CONFIGURED_RUN == run_directory:
        return
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[stage]}</cyan> | {message}",
    )
    logger.add(
        run_directory / "logs" / "run.log",
        level="DEBUG",
        rotation="50 MB",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[stage]} | {message}",
    )
    logger.add(
        run_directory / "logs" / "error.log",
        level="ERROR",
        rotation="50 MB",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[stage]} | {message}\n{exception}",
    )
    logger.configure(extra={"stage": "-"})
    _CONFIGURED_RUN = run_directory


def get_stage_logger(stage_name: str):
    """Return a logger bound to a stage name so console/file lines are prefixed with it."""
    return logger.bind(stage=stage_name)
