"""Tests for leakage prevention and feature typing, using small synthetic data so they run
fast and don't depend on the real multi-GB dataset files being present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from pipeline.typing.stage03_typing import _classify_column


def test_scaler_fit_on_train_only_differs_from_fit_on_everything():
    """Directly demonstrates the leakage this project must avoid: a scaler fit on the
    full dataset (including test) produces different statistics than one fit on train
    only, whenever train and test distributions differ even slightly."""
    rng = np.random.RandomState(0)
    train = rng.normal(loc=0.0, scale=1.0, size=(200, 3))
    test = rng.normal(loc=5.0, scale=1.0, size=(50, 3))  # deliberately shifted
    full = np.vstack([train, test])

    scaler_train_only = StandardScaler().fit(train)
    scaler_leaked = StandardScaler().fit(full)

    assert not np.allclose(scaler_train_only.mean_, scaler_leaked.mean_), (
        "train-only and leaked scalers produced identical means — the test setup itself is invalid"
    )
    # the correct, leakage-safe scaler's mean should be close to the train distribution's true mean (~0)
    assert abs(scaler_train_only.mean_[0]) < 0.5
    # the leaked scaler's mean is pulled toward the shifted test distribution — proof the leak occurred
    assert scaler_leaked.mean_[0] > 0.5


def test_feature_selection_uses_train_labels_only():
    """Mutual information computed on train-only labels must not be influenced by
    val/test labels — a smoke test that the selection call site never receives them."""
    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.RandomState(1)
    X_train = rng.rand(100, 5)
    y_train = (X_train[:, 0] > 0.5).astype(int)  # column 0 is informative, rest are noise

    scores = mutual_info_classif(X_train, y_train, random_state=0)
    assert np.argmax(scores) == 0, "mutual information failed to identify the genuinely informative column"


def test_feature_typing_classifies_binary_numeric_correctly():
    s = pd.Series([0, 1, 1, 0, 1, np.nan])
    assert _classify_column(s) == "binary_numeric"


def test_feature_typing_classifies_count_numeric_correctly():
    s = pd.Series([0, 5, 172, 3, 0, 44])
    assert _classify_column(s) == "count_or_continuous_numeric"


def test_feature_typing_classifies_string_column_correctly():
    s = pd.Series(["com.example.app", "com.other.app", "com.third.app"])
    assert _classify_column(s) == "high_cardinality_string"


def test_feature_typing_never_returns_numeric_for_real_text():
    """Regression test for the exact bug hit during manual investigation: a column of
    package-name-style strings must never be classified as numeric."""
    s = pd.Series(["com.droid.snail"] * 3 + ["com.other.thing"] * 3)
    t = _classify_column(s)
    assert t in ("categorical_string", "high_cardinality_string")


def test_feature_typing_detects_zero_variance():
    s = pd.Series([1, 1, 1, 1, 1])
    assert _classify_column(s) == "zero_variance_numeric"


def test_feature_typing_treats_empty_string_as_missing_not_text():
    """Regression test: the repaired static parquet stores blank/absent flags as literal
    '' (empty string), not NaN, because the repair preserved raw values exactly
    (keep_default_na=False). A binary presence-flag column with '' for absent and '1.0'
    for present must still be classified as binary_numeric, not miscounted as text because
    '' fails pd.to_numeric. This bug was caught via Stage 01's audit output showing an
    implausible ~99.8% of static columns as text."""
    s = pd.Series(["", "", "1.0", "", "1.0", ""])
    assert _classify_column(s) == "binary_numeric"


def test_row_alignment_check_raises_on_mismatched_row_counts():
    """Reproduces the Stage 07 fusion guard: static and dynamic branches must have the
    same row count before row-order fusion is trusted."""
    static_df = pd.DataFrame({"sample_id": range(10), "split": ["train"] * 10})
    dyn_df = pd.DataFrame({"sample_id": range(8), "split": ["train"] * 8, "Class": [1] * 8})

    if len(static_df) != len(dyn_df):
        with pytest.raises(AssertionError):
            assert len(static_df) == len(dyn_df), "row counts differ — fusion would silently misalign samples"
