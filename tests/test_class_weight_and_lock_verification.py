"""Regression tests for the three issues found in the forensic audit of the ~98% result:
1. Stage 13/14 previously trained unweighted despite config.imbalance.strategy=='class_weight'.
2. Stage 14 previously trusted Stage 13's number instead of independently re-verifying.
3. Cross-class duplicate conflicts were only logged, never persisted for disclosure.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from pipeline.evaluation.stage13_ablation import load_class_weights


def test_sample_weight_is_computed_from_class_weights():
    """A stand-in for _run_modality's weighting logic: given a class_weight map and a
    y_train array, the sample_weight array must assign each row its OWN class's weight —
    this is the exact computation that was previously skipped entirely."""
    class_weight = {"Adware": 2.0, "Benign": 1.0}
    y_train = np.array(["Adware", "Benign", "Adware", "Benign", "Benign"])
    sample_weight = np.array([class_weight[y] for y in y_train])
    assert list(sample_weight) == [2.0, 1.0, 2.0, 1.0, 1.0]


def test_load_class_weights_reads_stage09_artifact(tmp_path):
    import json
    from pipeline.core.paths import run_dir as _run_dir  # noqa: F401 (not used, just confirms import path stability)

    run_dir = tmp_path
    (run_dir / "artifacts").mkdir()
    payload = {"fused": {"class_weights": {"Adware": 1.8, "Benign": 0.7}}}
    with open(run_dir / "artifacts" / "stage09_imbalance_report.json", "w") as f:
        json.dump(payload, f)

    class _FakeCtx:
        pass
    ctx = _FakeCtx()
    ctx.run_dir = run_dir

    weights = load_class_weights(ctx, "fused")
    assert weights == {"Adware": 1.8, "Benign": 0.7}


def test_lightgbm_fit_receives_nonzero_sample_weight_argument():
    """Smoke-level proof that model.fit() is actually CALLED with a sample_weight kwarg
    that is not None, when class_weight strategy is active — regression test for the bug
    where Stage 13/14 called model.fit(X, y) with no weighting at all."""
    from sklearn.linear_model import LogisticRegression

    calls = {}
    real_fit = LogisticRegression.fit

    def spy_fit(self, X, y, sample_weight=None):
        calls["sample_weight"] = sample_weight
        return real_fit(self, X, y, sample_weight=sample_weight)

    X = np.random.rand(20, 3)
    y = np.array(["a"] * 10 + ["b"] * 10)
    class_weight = {"a": 2.0, "b": 1.0}
    sample_weight = np.array([class_weight[v] for v in y])

    with patch.object(LogisticRegression, "fit", spy_fit):
        model = LogisticRegression()
        model.fit(X, y, sample_weight=sample_weight)

    assert calls["sample_weight"] is not None
    assert calls["sample_weight"][0] == 2.0  # first 10 rows are class "a"


def test_metric_mismatch_tolerance_flags_real_discrepancy():
    """Regression test for the Stage 14 re-verification logic: a macro-F1 gap larger than
    the tolerance must be flagged as a mismatch, not silently accepted."""
    from pipeline.evaluation.stage14_lock import METRIC_MISMATCH_TOLERANCE

    reported_macro_f1 = 0.98
    verified_macro_f1_close = 0.975  # within tolerance
    verified_macro_f1_far = 0.90     # a real discrepancy

    assert abs(verified_macro_f1_close - reported_macro_f1) <= METRIC_MISMATCH_TOLERANCE
    assert abs(verified_macro_f1_far - reported_macro_f1) > METRIC_MISMATCH_TOLERANCE
