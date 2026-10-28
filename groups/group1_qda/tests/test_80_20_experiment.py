"""Tests for Group 1 QDA 80:20 Stratified Experiment.

Covers:
1. Stratified 80:20 train/test splitting and leakage prevention
2. Leakage-safe preprocessing (fitted on train only)
3. QDA regularization handling collinearity
4. Evaluation metric calculation and consistency
"""

import unittest
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from groups.group1_qda.src.run_80_20_experiment import (
    create_stratified_80_20_split,
    fit_preprocessing,
    apply_preprocessing,
    compute_test_metrics,
)


class TestQDA8020Experiment(unittest.TestCase):
    """Test suite for the 80:20 QDA experiment pipeline."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.n_samples = 100
        self.classes = ["Adware", "Banking", "Benign", "Riskware", "SMS Malware"]
        labels = self.rng.choice(self.classes, size=self.n_samples, p=[0.1, 0.2, 0.15, 0.25, 0.3])
        self.sample_ids = [f"sample_{i:04d}" for i in range(self.n_samples)]
        self.manifest = pd.DataFrame({
            "sample_id": self.sample_ids,
            "label_canonical": labels,
        })
        self.feature_names = [f"feat_{i}" for i in range(10)]
        self.features = pd.DataFrame(
            self.rng.randn(self.n_samples, 10),
            columns=self.feature_names,
            index=self.sample_ids,
        )

    def test_stratified_80_20_split_proportions(self):
        """Split should maintain ~80:20 ratio and zero overlap between sets."""
        train_ids, test_ids = create_stratified_80_20_split(
            self.manifest,
            label_column="label_canonical",
            id_column="sample_id",
            seed=42,
        )

        self.assertEqual(len(train_ids) + len(test_ids), self.n_samples)
        self.assertEqual(len(train_ids), 80)
        self.assertEqual(len(test_ids), 20)

        # Zero overlap
        intersection = set(train_ids).intersection(set(test_ids))
        self.assertEqual(len(intersection), 0, "Train and test IDs must have zero overlap")

    def test_stratified_split_preserves_classes(self):
        """All classes present in the manifest should be represented in train and test."""
        train_ids, test_ids = create_stratified_80_20_split(
            self.manifest,
            label_column="label_canonical",
            id_column="sample_id",
            seed=42,
        )
        labels_by_id = dict(zip(self.manifest["sample_id"], self.manifest["label_canonical"]))
        train_labels = {labels_by_id[i] for i in train_ids}
        test_labels = {labels_by_id[i] for i in test_ids}
        self.assertEqual(train_labels, set(self.classes))
        self.assertEqual(test_labels, set(self.classes))

    def test_leakage_safe_preprocessing(self):
        """Scaler and imputer must fit on train only; test must be transformed using train params."""
        train_ids = self.sample_ids[:80]
        test_ids = self.sample_ids[80:]

        X_train = self.features.loc[train_ids].values.copy()
        X_test = self.features.loc[test_ids].values.copy()

        # Inject some NaNs into train and test
        X_train[0, 0] = np.nan
        X_test[0, 0] = np.nan

        preproc = fit_preprocessing(X_train, self.feature_names)
        X_train_proc = preproc["X_train_processed"]
        X_test_proc = apply_preprocessing(X_test, preproc)

        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertFalse(np.isnan(X_test_proc).any())
        self.assertEqual(X_train_proc.shape, (80, 10))
        self.assertEqual(X_test_proc.shape, (20, 10))

        # Check mean of train scaled is ~0
        np.testing.assert_allclose(X_train_proc.mean(axis=0), 0.0, atol=1e-5)

    def test_qda_regularization_handles_collinearity(self):
        """Unregularized QDA can fail on collinear data; reg_param > 0 stabilizes it."""
        n = 50
        X = np.zeros((n, 5))
        X[:, 0] = np.linspace(-1, 1, n)
        X[:, 1] = 2.0 * X[:, 0]  # Exact collinearity
        X[:, 2:] = self.rng.randn(n, 3) * 0.01
        y = np.array(["A"] * 25 + ["B"] * 25)

        qda_reg = QuadraticDiscriminantAnalysis(reg_param=0.1)
        qda_reg.fit(X, y)
        preds = qda_reg.predict(X)
        self.assertEqual(len(preds), n)

    def test_evaluation_metrics_structure(self):
        """Evaluate model returns required metrics dictionary with all expected keys."""
        train_ids = self.sample_ids[:80]
        test_ids = self.sample_ids[80:]

        X_train = self.features.loc[train_ids].values
        y_train = self.manifest.set_index("sample_id").loc[train_ids, "label_canonical"].values
        X_test = self.features.loc[test_ids].values
        y_test = self.manifest.set_index("sample_id").loc[test_ids, "label_canonical"].values

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(X_train, y_train)

        class_names = sorted(list(set(y_train)))
        y_pred = clf.predict(X_test)
        metrics = compute_test_metrics(y_test, y_pred, class_names)

        required_keys = [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "per_class_report",
            "confusion_matrix",
            "test_sample_count",
        ]
        for key in required_keys:
            self.assertIn(key, metrics, f"Missing key: {key}")

        self.assertEqual(metrics["test_sample_count"], len(test_ids))
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
