"""Group 1 QDA -- Complete 80:20 stratified experiment with memory-efficient loading.

This standalone script implements the full QDA experiment pipeline:
1. Memory-efficient loading of only the selected K=50 features from parquet
2. 80:20 stratified train/test splitting (using all 11,598 samples)
3. Leakage-safe preprocessing (imputation + scaling fitted on train only)
4. QDA model training with reg_param selection via cross-validation on train
5. Evaluation on held-out 20% test set
6. Saving model artifacts, metrics, plots, and documentation

It does NOT use the existing data.py load_dataset() (which loads the full 63K-column
table and OOMs on low-memory machines). Instead, it uses PyArrow column projection
to read only the 50 needed features + sample_id, keeping peak memory under 100 MB.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import yaml
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Import the adapter to build the QDA model (keeping existing adapter unchanged)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.adapters import qda as qda_adapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
K_FEATURES = 50

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the 80:20 experiment YAML configuration."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root_setting = config.get("project_root", ".")
    project_root = (path.parent / root_setting).resolve()
    config["_resolved_project_root"] = project_root
    config["_config_path"] = path
    return config


def _resolve(value: str, root: Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(value).expanduser()
    resolved = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Memory-Efficient Data Loading
# ---------------------------------------------------------------------------

def load_selected_features_from_parquet(
    parquet_path: Path,
    selected_scaled_columns: list[str],
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Load only the selected feature columns + ID from a large Parquet file.

    This avoids reading the full 63K-column table. Peak memory is proportional
    to n_samples x (K + 1) instead of n_samples x 63K.
    """
    pf = pq.ParquetFile(str(parquet_path), memory_map=True)
    schema_names = set(pf.schema_arrow.names)

    if id_column not in schema_names:
        raise ValueError(f"ID column '{id_column}' not found in parquet schema")

    missing = [c for c in selected_scaled_columns if c not in schema_names]
    if missing:
        raise ValueError(f"Missing feature columns in parquet: {missing[:10]}")

    columns_to_read = [id_column] + selected_scaled_columns
    table = pf.read(columns=columns_to_read, use_threads=False)
    df = table.to_pandas()
    del table
    gc.collect()

    df = df.set_index(id_column)
    print(f"  Loaded {len(df)} samples x {len(df.columns)} features "
          f"({df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)")
    return df


def load_feature_selection_report(
    selection_path: Path,
    modality: str,
    method: str,
    k: int,
) -> list[str]:
    """Extract the top-K feature names from the feature selection report."""
    with selection_path.open(encoding="utf-8") as f:
        data = json.load(f)
    try:
        base_names = data[modality]["methods"][method][str(k)]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot find features for {modality}/{method}/K={k} in {selection_path}"
        ) from exc
    if not isinstance(base_names, list) or len(base_names) != k:
        raise ValueError(f"Expected {k} features, got {len(base_names)} in {selection_path}")
    return base_names


# ---------------------------------------------------------------------------
# Stratified 80:20 Splitting
# ---------------------------------------------------------------------------

def create_stratified_80_20_split(
    manifest: pd.DataFrame,
    label_column: str,
    id_column: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified 80:20 train/test split.

    Uses ALL samples from the manifest (ignoring the original train/val/test
    assignments) and re-splits them 80:20 with stratification on label_column.

    Returns (train_ids, test_ids) as numpy arrays.
    """
    from sklearn.model_selection import train_test_split

    ids = manifest[id_column].to_numpy()
    labels = manifest[label_column].to_numpy()

    train_ids, test_ids, train_labels, test_labels = train_test_split(
        ids, labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )

    # Validate stratification
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    total_dist = Counter(labels)

    print(f"\n  80:20 Stratified Split Summary:")
    print(f"  {'Class':<15} {'Total':>6} {'Train':>6} {'Test':>6} {'Train%':>7} {'Test%':>7}")
    print(f"  {'-'*58}")
    for cls in sorted(total_dist.keys()):
        total = total_dist[cls]
        tr = train_dist.get(cls, 0)
        te = test_dist.get(cls, 0)
        print(f"  {cls:<15} {total:>6} {tr:>6} {te:>6} {100*tr/total:>6.1f}% {100*te/total:>6.1f}%")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<15} {len(ids):>6} {len(train_ids):>6} {len(test_ids):>6} "
          f"{100*len(train_ids)/len(ids):>6.1f}% {100*len(test_ids)/len(ids):>6.1f}%")

    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Leakage-Safe Preprocessing
# ---------------------------------------------------------------------------

def fit_preprocessing(
    X_train: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Fit imputer and scaler on training data only.

    Returns a preprocessing dict that can be applied to test data.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    return {
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_train_processed": X_train_scaled,
    }


def apply_preprocessing(
    X: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """Apply fitted preprocessing to new data (test set)."""
    X_imputed = preprocessing["imputer"].transform(X)
    return preprocessing["scaler"].transform(X_imputed)


# ---------------------------------------------------------------------------
# QDA Model Building and Training
# ---------------------------------------------------------------------------

def build_qda_model(reg_param: float, max_features: int) -> QuadraticDiscriminantAnalysis:
    """Build QDA model using the existing pipeline adapter."""
    adapter_config = {
        "models": {
            "qda": {
                "reg_param": reg_param,
                "max_features": max_features,
            }
        }
    }
    return qda_adapter.build_model(adapter_config)


def select_best_reg_param(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: list[float],
    max_features: int,
    seed: int,
    n_folds: int = 5,
) -> tuple[float, list[dict[str, Any]]]:
    """Select the best reg_param via stratified k-fold cross-validation on training data.

    Uses macro F1 as the selection metric.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []

    for reg_param in candidates:
        model = build_qda_model(reg_param, max_features)
        caught: list[Any] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="f1_macro",
                    error_score="raise",
                )
            captured_warnings = [str(w.message) for w in caught]
            result = {
                "reg_param": reg_param,
                "status": "PASS",
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
                "cv_macro_f1_folds": scores.tolist(),
                "warnings": captured_warnings,
            }
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            captured_warnings = [str(w.message) for w in caught] if caught else []
            result = {
                "reg_param": reg_param,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": captured_warnings,
            }
        results.append(result)
        status_marker = "[OK]" if result["status"] == "PASS" else "[FAIL]"
        cv_info = (f"CV F1={result['cv_macro_f1_mean']:.4f}+/-{result['cv_macro_f1_std']:.4f}"
                   if result["status"] == "PASS" else result.get("error", "FAIL"))
        print(f"  {status_marker} reg_param={reg_param:.2f}: {cv_info}")

    passed = [r for r in results if r["status"] == "PASS"]
    if not passed:
        raise RuntimeError(
            "All QDA reg_param candidates failed during cross-validation. "
            f"Details: {[r.get('error', 'unknown') for r in results]}"
        )

    best = max(passed, key=lambda r: r["cv_macro_f1_mean"])
    print(f"\n  >> Best reg_param: {best['reg_param']}")
    return best["reg_param"], results


# ---------------------------------------------------------------------------
# Evaluation Metrics and Reporting
# ---------------------------------------------------------------------------

def compute_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    """Compute comprehensive test-set metrics."""
    report = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_labels,
        "test_sample_count": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    title: str = "QDA Confusion Matrix (80:20 Test Split)",
) -> None:
    """Save a publication-quality confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix: {output_path}")


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """Save a row-normalized confusion matrix (recall per class)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels, ax=ax,
        vmin=0, vmax=1, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("QDA Normalized Confusion Matrix (Recall)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved normalized confusion matrix: {output_path}")


def plot_per_class_f1(
    report: dict[str, Any],
    labels: list[str],
    output_path: Path,
) -> None:
    """Save a per-class F1-score bar chart."""
    f1_scores = [report.get(label, {}).get("f1-score", 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.barh(labels, f1_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores (QDA, 80:20 Split)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per-class F1 chart: {output_path}")


def plot_reg_param_search(
    cv_results: list[dict[str, Any]],
    best_param: float,
    output_path: Path,
) -> None:
    """Plot regularization parameter search results."""
    passed = [r for r in cv_results if r["status"] == "PASS"]
    if not passed:
        return

    params = [r["reg_param"] for r in passed]
    means = [r["cv_macro_f1_mean"] for r in passed]
    stds = [r["cv_macro_f1_std"] for r in passed]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(params, means, yerr=stds, fmt="o-", capsize=5,
                color="steelblue", markersize=8, linewidth=2)
    best_idx = params.index(best_param) if best_param in params else None
    if best_idx is not None:
        ax.plot(params[best_idx], means[best_idx], "r*", markersize=20, zorder=5,
                label=f"Best: reg_param={best_param:.2f}")
    ax.set_xlabel("Regularization Parameter", fontsize=12)
    ax.set_ylabel("CV Macro F1 Score", fontsize=12)
    ax.set_title("QDA Regularization Parameter Search", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reg_param search plot: {output_path}")


def plot_class_distribution(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the class distribution in train and test sets."""
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    classes = sorted(set(train_counts) | set(test_counts))

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    train_vals = [train_counts.get(c, 0) for c in classes]
    test_vals = [test_counts.get(c, 0) for c in classes]
    bars1 = ax.bar(x - width/2, train_vals, width, label="Train (80%)",
                   color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_vals, width, label="Test (20%)",
                   color="coral", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Malware Class", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Class Distribution: 80:20 Stratified Split", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.legend(fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved class distribution plot: {output_path}")


# ---------------------------------------------------------------------------
# Package Versions
# ---------------------------------------------------------------------------

def package_versions() -> dict[str, str]:
    """Collect installed package versions for reproducibility."""
    pkgs = ("numpy", "pandas", "scikit-learn", "pyarrow", "PyYAML",
            "joblib", "matplotlib", "seaborn")
    result = {}
    for pkg in pkgs:
        try:
            result[pkg] = version(pkg)
        except PackageNotFoundError:
            result[pkg] = "not-installed"
    result["python"] = platform.python_version()
    return result


# ---------------------------------------------------------------------------
# File Hashing
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(config_path: str | Path) -> Path:
    """Execute the complete 80:20 QDA experiment."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"=" * 70)
    print(f"  Group 1 QDA -- 80:20 Stratified Experiment")
    print(f"  Started: {timestamp}")
    print(f"=" * 70)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    print(f"\n[1/8] Loading configuration...")
    config = load_experiment_config(config_path)
    root = config["_resolved_project_root"]
    seed = config.get("random_seed", RANDOM_SEED)
    k = config["feature_selection"]["k"]
    method = config["feature_selection"]["method"]
    modality = config["dataset"]["feature_modality"]
    reg_param_candidates = config["qda"]["reg_param_candidates"]

    feature_path = _resolve(config["dataset"]["feature_path"], root)
    label_path = _resolve(config["dataset"]["label_source"], root)
    selection_source = config["feature_selection"].get("selection_source", "")
    selection_path = _resolve(selection_source, root) if selection_source else (
        feature_path.parent / "stage08_feature_selection.json"
    )

    print(f"  Feature file: {feature_path}")
    print(f"  Manifest: {label_path}")
    print(f"  Feature selection: {selection_path}")
    print(f"  Modality: {modality}, Method: {method}, K: {k}")
    print(f"  Random seed: {seed}")
    print(f"  reg_param candidates: {reg_param_candidates}")

    # ------------------------------------------------------------------
    # 2. Load feature selection report and determine columns
    # ------------------------------------------------------------------
    print(f"\n[2/8] Loading feature selection report...")
    base_feature_names = load_feature_selection_report(selection_path, modality, method, k)
    scaled_columns = [f"{name}__scaled" for name in base_feature_names]
    print(f"  Top {k} features loaded (e.g.: {base_feature_names[:3]}...)")

    # ------------------------------------------------------------------
    # 3. Load manifest and create 80:20 stratified split
    # ------------------------------------------------------------------
    print(f"\n[3/8] Creating 80:20 stratified split...")
    id_col = config["dataset"]["sample_id_column"]
    label_col = config["dataset"]["label_column"]
    manifest = pd.read_parquet(label_path)

    # Validate manifest
    assert id_col in manifest.columns, f"ID column '{id_col}' not in manifest"
    assert label_col in manifest.columns, f"Label column '{label_col}' not in manifest"
    assert manifest[id_col].is_unique, "Manifest has duplicate sample IDs"
    assert not manifest[label_col].isna().any(), "Manifest has missing labels"

    # Validate allowed labels if configured
    allowed = config["dataset"].get("allowed_labels")
    if allowed:
        unexpected = set(manifest[label_col].unique()) - set(allowed)
        if unexpected:
            raise ValueError(f"Unexpected labels in manifest: {unexpected}")

    train_ids, test_ids = create_stratified_80_20_split(manifest, label_col, id_col, seed)

    # Create lookup dictionaries
    label_by_id = dict(zip(manifest[id_col], manifest[label_col]))
    y_train = np.array([label_by_id[sid] for sid in train_ids])
    y_test = np.array([label_by_id[sid] for sid in test_ids])

    # ------------------------------------------------------------------
    # 4. Load selected features (memory-efficient)
    # ------------------------------------------------------------------
    print(f"\n[4/8] Loading {k} selected features (memory-efficient)...")
    feature_df = load_selected_features_from_parquet(feature_path, scaled_columns, id_col)

    # Split into train/test feature matrices
    X_train_df = feature_df.loc[train_ids]
    X_test_df = feature_df.loc[test_ids]
    assert len(X_train_df) == len(train_ids), "Train ID mismatch"
    assert len(X_test_df) == len(test_ids), "Test ID mismatch"

    X_train_raw = X_train_df.values.astype(np.float64)
    X_test_raw = X_test_df.values.astype(np.float64)
    del feature_df, X_train_df, X_test_df
    gc.collect()

    # Check for infinities
    if np.isinf(X_train_raw).any() or np.isinf(X_test_raw).any():
        raise ValueError("Feature data contains infinite values")

    print(f"  Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
    print(f"  Missing in train: {np.isnan(X_train_raw).sum()}, "
          f"Missing in test: {np.isnan(X_test_raw).sum()}")

    # ------------------------------------------------------------------
    # 5. Fit preprocessing on training data only (leakage-safe)
    # ------------------------------------------------------------------
    print(f"\n[5/8] Fitting preprocessing (train-only, leakage-safe)...")
    preprocessing = fit_preprocessing(X_train_raw, scaled_columns)
    X_train = preprocessing["X_train_processed"]
    X_test = apply_preprocessing(X_test_raw, preprocessing)
    del X_train_raw, X_test_raw
    gc.collect()

    print(f"  Imputer fitted on {len(X_train)} train samples")
    print(f"  Scaler fitted on {len(X_train)} train samples")
    print(f"  Train scaled shape: {X_train.shape}")
    print(f"  Test scaled shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 6. Select best reg_param via cross-validation on training data
    # ------------------------------------------------------------------
    print(f"\n[6/8] Selecting reg_param via {5}-fold stratified CV on training data...")
    best_reg_param, cv_results = select_best_reg_param(
        X_train, y_train, reg_param_candidates, k, seed
    )
    print(f"\n  >> Best reg_param: {best_reg_param}")

    # ------------------------------------------------------------------
    # 7. Train final model on full training set with best reg_param
    # ------------------------------------------------------------------
    print(f"\n[7/8] Training final QDA model (reg_param={best_reg_param})...")
    final_model = build_qda_model(best_reg_param, k)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(X_train, y_train)
    training_warnings = [str(w.message) for w in caught]
    if training_warnings:
        print(f"  [WARN] Training warnings: {training_warnings}")

    train_pred = final_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average="macro", zero_division=0)
    print(f"  Train accuracy: {train_accuracy:.4f}, Train macro-F1: {train_f1:.4f}")

    # ------------------------------------------------------------------
    # 8. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print(f"\n[8/8] Evaluating on held-out test set ({len(y_test)} samples)...")
    y_pred = final_model.predict(X_test)
    class_labels = sorted(final_model.classes_.tolist())
    test_metrics = compute_test_metrics(y_test, y_pred, class_labels)

    print(f"\n  +{'-'*50}+")
    print(f"  | TEST SET RESULTS (80:20 Stratified Split)       |")
    print(f"  +{'-'*50}+")
    print(f"  | Accuracy:           {test_metrics['accuracy']:.4f}{' '*25}|")
    print(f"  | Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}{' '*25}|")
    print(f"  | Macro Precision:    {test_metrics['macro_precision']:.4f}{' '*25}|")
    print(f"  | Macro Recall:       {test_metrics['macro_recall']:.4f}{' '*25}|")
    print(f"  | Macro F1:           {test_metrics['macro_f1']:.4f}{' '*25}|")
    print(f"  | Weighted F1:        {test_metrics['weighted_f1']:.4f}{' '*25}|")
    print(f"  | Test Samples:       {test_metrics['test_sample_count']}{' '*29}|")
    print(f"  +{'-'*50}+")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_base = config.get("output_dir", "groups/group1_qda/outputs/qda_80_20_experiment")
    output_base_path = Path(output_base)
    if not output_base_path.is_absolute():
        output_base_path = root / output_base_path

    run_id = f"run_{timestamp}"
    run_dir = (output_base_path / run_id).resolve()

    # Safety: never write into runs/real/
    protected = (root / "runs" / "real").resolve()
    try:
        run_dir.relative_to(protected)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"Refusing to write into protected directory: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir()

    print(f"\n  Saving outputs to: {run_dir}")

    # Save model
    model_payload = {
        "model": final_model,
        "preprocessing": {
            "imputer": preprocessing["imputer"],
            "scaler": preprocessing["scaler"],
            "feature_names": preprocessing["feature_names"],
        },
        "classes": class_labels,
        "experiment_type": "80_20_stratified",
        "random_seed": seed,
    }
    joblib.dump(model_payload, run_dir / "qda_model.joblib")
    print(f"  Saved model: qda_model.joblib")

    # Save split manifest
    split_df = pd.DataFrame({
        id_col: np.concatenate([train_ids, test_ids]),
        "split_80_20": (["train"] * len(train_ids)) + (["test"] * len(test_ids)),
        label_col: np.concatenate([y_train, y_test]),
    })
    split_df.to_parquet(run_dir / "experiment_split_manifest.parquet", index=False)
    split_df.to_csv(run_dir / "experiment_split_manifest.csv", index=False)
    print(f"  Saved split manifest (parquet + csv)")

    # Save test metrics
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    print(f"  Saved test_metrics.json")

    # Save training metadata
    metadata = {
        "experiment_type": "group1_qda_80_20_stratified",
        "timestamp": timestamp,
        "config_path": str(config["_config_path"]),
        "feature_path": str(feature_path),
        "label_source": str(label_path),
        "selection_source": str(selection_path),
        "feature_modality": modality,
        "feature_selection_method": method,
        "feature_selection_k": k,
        "selected_feature_base_names": base_feature_names,
        "selected_feature_scaled_columns": scaled_columns,
        "random_seed": seed,
        "split_strategy": "stratified_80_20",
        "test_size": TEST_SIZE,
        "train_sample_count": int(len(train_ids)),
        "test_sample_count": int(len(test_ids)),
        "total_sample_count": int(len(train_ids) + len(test_ids)),
        "train_label_distribution": dict(Counter(y_train.tolist())),
        "test_label_distribution": dict(Counter(y_test.tolist())),
        "reg_param_candidates": reg_param_candidates,
        "selected_reg_param": best_reg_param,
        "cv_results": cv_results,
        "training_warnings": training_warnings,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_f1,
        "test_metrics": test_metrics,
        "class_labels": class_labels,
        "package_versions": package_versions(),
        "input_file_hashes": {
            "feature_parquet_sha256": file_sha256(feature_path),
            "manifest_sha256": file_sha256(label_path),
            "selection_report_sha256": file_sha256(selection_path),
        },
    }
    with (run_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved experiment_metadata.json")

    # Save selected features
    with (run_dir / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "k": k,
            "base_feature_names": base_feature_names,
            "scaled_column_names": scaled_columns,
        }, f, indent=2)
    print(f"  Saved selected_features.json")

    # Generate plots
    print(f"\n  Generating plots...")
    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_labels, plots_dir / "confusion_matrix.png")
    plot_normalized_confusion_matrix(cm, class_labels, plots_dir / "confusion_matrix_normalized.png")
    plot_per_class_f1(test_metrics["per_class_report"], class_labels, plots_dir / "per_class_f1.png")
    plot_reg_param_search(cv_results, best_reg_param, plots_dir / "reg_param_search.png")
    plot_class_distribution(y_train, y_test, plots_dir / "class_distribution.png")

    # Save per-class results as CSV for easy viewing
    per_class_rows = []
    for cls in class_labels:
        cls_metrics = test_metrics["per_class_report"].get(cls, {})
        per_class_rows.append({
            "class": cls,
            "precision": cls_metrics.get("precision", 0),
            "recall": cls_metrics.get("recall", 0),
            "f1_score": cls_metrics.get("f1-score", 0),
            "support": cls_metrics.get("support", 0),
        })
    pd.DataFrame(per_class_rows).to_csv(run_dir / "per_class_metrics.csv", index=False)
    print(f"  Saved per_class_metrics.csv")

    print(f"\n{'=' * 70}")
    print(f"  [OK] Experiment complete! All outputs saved to:")
    print(f"    {run_dir}")
    print(f"{'=' * 70}")

    return run_dir


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Group 1 QDA -- 80:20 Stratified Experiment Runner"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "qda_static_k50_80_20.yaml"
        ),
        help="Path to the 80:20 experiment YAML configuration.",
    )
    args = parser.parse_args(argv)

    try:
        run_dir = run_experiment(args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nExperiment stopped: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
