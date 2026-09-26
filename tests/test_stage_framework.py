"""Tests for the core resumability / leakage-prevention framework.

These are foundational and must pass before any expensive model training is trusted —
they test the mechanism itself (skip-if-succeeded, hard-stop-on-missing-dependency,
hard-stop-on-failure with checkpoint preserved), not any particular dataset result.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from pipeline.core.stage import Stage, StageContext, StageDependencyError, StageExecutionError
from pipeline.core.status import get_stage_record


@pytest.fixture
def tmp_run_dir():
    d = Path(tempfile.mkdtemp())
    for sub in ["logs", "artifacts", "checkpoints", "metrics", "plots", "diagrams", "manifests", "reports"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


class _OKStage(Stage):
    name = "ok_stage"
    depends_on: list[str] = []

    def config_section(self, cfg):
        return {"x": cfg.get("x", 1)}

    def run(self, ctx):
        return {"result": "ok"}


class _FailingStage(Stage):
    name = "failing_stage"
    depends_on: list[str] = []

    def config_section(self, cfg):
        return {}

    def run(self, ctx):
        raise ValueError("intentional failure for test")


class _DependentStage(Stage):
    name = "dependent_stage"
    depends_on = ["ok_stage"]

    def config_section(self, cfg):
        return {}

    def run(self, ctx):
        return {"result": "ran after dependency"}


def test_stage_succeeds_and_records_status(tmp_run_dir):
    ctx = StageContext(cfg={"x": 1}, run_dir=tmp_run_dir)
    record = _OKStage().execute(ctx)
    assert record["status"] == "SUCCESS"
    stored = get_stage_record(tmp_run_dir, "ok_stage")
    assert stored["status"] == "SUCCESS"


def test_stage_is_skipped_when_already_succeeded_with_same_config(tmp_run_dir):
    ctx = StageContext(cfg={"x": 1}, run_dir=tmp_run_dir)
    _OKStage().execute(ctx)
    calls = {"n": 0}

    class _CountingStage(_OKStage):
        def run(self, c):
            calls["n"] += 1
            return {"result": "ok"}

    _CountingStage().execute(ctx)
    assert calls["n"] == 0, "stage re-ran even though it already succeeded with matching config"


def test_stage_reruns_when_forced(tmp_run_dir):
    ctx = StageContext(cfg={"x": 1}, run_dir=tmp_run_dir, force_stages={"ok_stage"})
    _OKStage().execute(ctx)
    calls = {"n": 0}

    class _CountingStage(_OKStage):
        def run(self, c):
            calls["n"] += 1
            return {"result": "ok"}

    _CountingStage().execute(ctx)
    assert calls["n"] == 1, "--force-stage did not trigger a re-run"


def test_stage_reruns_when_config_changes(tmp_run_dir):
    ctx1 = StageContext(cfg={"x": 1}, run_dir=tmp_run_dir)
    _OKStage().execute(ctx1)
    ctx2 = StageContext(cfg={"x": 2}, run_dir=tmp_run_dir)  # different config -> different hash
    calls = {"n": 0}

    class _CountingStage(_OKStage):
        def run(self, c):
            calls["n"] += 1
            return {"result": "ok"}

    _CountingStage().execute(ctx2)
    assert calls["n"] == 1, "stage did not detect a changed config hash"


def test_dependent_stage_hard_stops_when_dependency_missing(tmp_run_dir):
    ctx = StageContext(cfg={}, run_dir=tmp_run_dir)
    with pytest.raises(StageDependencyError):
        _DependentStage().execute(ctx)


def test_dependent_stage_runs_after_dependency_succeeds(tmp_run_dir):
    ctx = StageContext(cfg={"x": 1}, run_dir=tmp_run_dir)
    _OKStage().execute(ctx)
    record = _DependentStage().execute(ctx)
    assert record["status"] == "SUCCESS"


def test_failed_stage_hard_stops_and_preserves_checkpoint(tmp_run_dir):
    ctx = StageContext(cfg={}, run_dir=tmp_run_dir)
    with pytest.raises(StageExecutionError):
        _FailingStage().execute(ctx)
    stored = get_stage_record(tmp_run_dir, "failing_stage")
    assert stored["status"] == "FAILED"
    assert "intentional failure" in stored["error"]
    assert "traceback" in stored


def test_downstream_stage_does_not_run_after_upstream_failure(tmp_run_dir):
    ctx = StageContext(cfg={}, run_dir=tmp_run_dir)
    with pytest.raises(StageExecutionError):
        _FailingStage().execute(ctx)

    class _DependsOnFailing(Stage):
        name = "depends_on_failing"
        depends_on = ["failing_stage"]

        def config_section(self, cfg):
            return {}

        def run(self, c):
            raise AssertionError("this must never execute")

    with pytest.raises(StageDependencyError):
        _DependsOnFailing().execute(ctx)
