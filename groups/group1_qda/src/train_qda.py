"""Train Group 1 QDA with preprocessing and feature selection fitted on train only."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from pipeline.adapters import qda as qda_adapter
from groups.group1_qda.src.data import (
    DataContractError,
    DatasetBundle,
    load_config,
    load_dataset,
)


def _feature_scores(
    method: str, X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> np.ndarray:
    if method == "mutual_info":
        return mutual_info_classif(X_train, y_train, random_state=seed)
    if method == "chi2":
        if np.any(X_train < 0):
            raise DataContractError(
                "chi2 feature selection requires non-negative training features. "
                "Choose mutual_info or use a non-negative feature representation."
            )
        scores, _ = chi2(X_train, y_train)
        return np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if method == "f_classif":
        scores, _ = f_classif(X_train, y_train)
        return np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if method == "lightgbm_importance":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise DataContractError(
                "feature_selection.method='lightgbm_importance' requires lightgbm."
            ) from exc
        model = LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1)
        model.fit(X_train, y_train)
        return model.feature_importances_.astype(float)
    raise DataContractError(
        f"Unsupported feature-selection method {method!r}. Supported methods: "
        "mutual_info, chi2, f_classif, lightgbm_importance."
    )


def _fit_preprocessing(
    bundle: DatasetBundle, config: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fs = config["feature_selection"]
    method = str(fs.get("method", ""))
    k = fs.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise DataContractError("feature_selection.k must be a positive integer.")

    train_mask = bundle.splits == "train"
    val_mask = bundle.splits == "validation"
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise DataContractError("Training and validation partitions must both be non-empty.")
    X_train_df = bundle.features.loc[train_mask]
    X_val_df = bundle.features.loc[val_mask]
    y_train = bundle.labels.loc[train_mask].to_numpy()
    y_val = bundle.labels.loc[val_mask].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_df)
    X_val_imputed = imputer.transform(X_val_df)
    if X_train_imputed.shape[1] != len(bundle.feature_schema):
        raise DataContractError(
            "Imputation changed the feature dimension. Check for all-missing training "
            "features or incompatible data."
        )

    variance_filter = VarianceThreshold(threshold=0.0)
    try:
        X_train_variable = variance_filter.fit_transform(X_train_imputed)

    except ValueError as exc:
        raise DataContractError(
            "No non-constant features remain in the official training partition."
        ) from exc
    variable_indices = np.flatnonzero(variance_filter.get_support())
    variable_names = [bundle.feature_schema[index] for index in variable_indices]
    if k > len(variable_names):
        raise DataContractError(
            f"Requested K={k}, but only {len(variable_names)} non-constant training "
            "features remain. Lower K or revise the feature dataset."
        )

    seed = config.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise DataContractError("random_seed must be an integer in [0, 2**32).")
    scores = _feature_scores(method, X_train_variable, y_train, seed)
    if len(scores) != len(variable_names) or not np.isfinite(scores).any():
        raise DataContractError(
            f"Feature selection method {method!r} did not produce usable feature scores."
        )
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    ranked_positions = np.lexsort((np.arange(len(scores)), -scores))
    selected_positions = ranked_positions[:k]
    selected_names = [variable_names[index] for index in selected_positions]
    selected_global_indices = variable_indices[selected_positions]

    scaler = StandardScaler()
    X_train_selected = X_train_imputed[:, selected_global_indices]
    X_val_selected = X_val_imputed[:, selected_global_indices]
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)

    preprocessing = {
        "imputer": imputer,
        "variance_filter": variance_filter,
        "selected_global_indices": selected_global_indices,
        "scaler": scaler,
        "input_feature_names": list(bundle.feature_schema),
        "selected_feature_names": selected_names,
    }
    training_class_counts = pd.Series(y_train).value_counts()
    if len(training_class_counts) < 2:
        raise DataContractError(
            "QDA training requires at least two distinct labels in the official training partition."
        )
    min_class_count = int(training_class_counts.min())
    warnings_report: list[str] = []
    if k >= min_class_count:
        warnings_report.append(
            f"K={k} is at least the smallest training-class count ({min_class_count}); "
            "QDA covariance estimates may be unstable. Inspect candidate fit warnings."
        )
    report = {
        "selection_method": method,
        "requested_k": k,
        "selected_k": int(len(selected_names)),
        "constant_features_removed_from_training": [
            name for name in bundle.feature_schema if name not in variable_names
        ],
        "selected_features": selected_names,
        "minimum_training_class_count": min_class_count,
        "warnings": warnings_report,
    }
    return preprocessing, X_train_scaled, X_val_scaled, y_train, y_val, report


def _transform(frame: pd.DataFrame, preprocessing: dict[str, Any]) -> np.ndarray:
    expected = preprocessing["input_feature_names"]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise DataContractError(
            f"Feature schema mismatch. Missing columns: {missing[:10]}; "
            f"unexpected columns: {extra[:10]}."
        )
    ordered = frame.loc[:, expected]
    imputed = preprocessing["imputer"].transform(ordered)
    selected = imputed[:, preprocessing["selected_global_indices"]]
    return preprocessing["scaler"].transform(selected)


def _build_model(config: dict[str, Any], reg_param: float) -> QuadraticDiscriminantAnalysis:
    adapter_config = {
        "models": {
            "qda": {
                "reg_param": reg_param,
                "max_features": int(config["feature_selection"]["k"]),
            }
        }
    }
    return qda_adapter.build_model(adapter_config)


def _new_run_directory(config: dict[str, Any], root: Path) -> Path:
    output_setting = Path(config["output_dir"])
    output_base = output_setting if output_setting.is_absolute() else root / output_setting
    requested_id = config.get("run_id")
    run_id = str(requested_id) if requested_id else datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    if not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise DataContractError("run_id must be a non-empty single directory name.")
    run_dir = (output_base / run_id).resolve()
    protected_group3_dir = (root / "runs" / "real").resolve()
    try:
        run_dir.relative_to(protected_group3_dir)
    except ValueError:
        pass
    else:
        raise DataContractError(
            f"Refusing to write Group 1 output under protected Group 3 artifacts: {run_dir}"
        )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing QDA run directory: {run_dir}. "
            "Choose a new run_id."
        ) from exc
    return run_dir


def _package_versions() -> dict[str, str]:
    packages = (
        "numpy", "pandas", "scikit-learn", "pyarrow", "PyYAML", "joblib", "lightgbm"
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    result["python"] = platform.python_version()
    return result


def train(config_path: str | Path) -> Path:
    config, resolved_config_path, root = load_config(config_path)
    bundle = load_dataset(config, root)
    preprocessing, X_train, X_val, y_train, y_val, selection_report = _fit_preprocessing(
        bundle, config
    )
    candidates = config["qda"].get("reg_param_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DataContractError("qda.reg_param_candidates must be a non-empty list.")
    for candidate in candidates:
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not np.isfinite(float(candidate))
            or not 0.0 <= float(candidate) <= 1.0
        ):
            raise DataContractError(
                f"Invalid QDA reg_param candidate {candidate!r}; values must be in [0, 1]."
            )

    candidate_results: list[dict[str, Any]] = []
    best: tuple[float, int, QuadraticDiscriminantAnalysis, dict[str, Any]] | None = None
    for index, candidate in enumerate(candidates):
        reg_param = float(candidate)
        model = _build_model(config, reg_param)
        caught: list[Any] = []
        captured_warnings: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
            captured_warnings = [str(item.message) for item in caught]
            macro_f1 = float(f1_score(y_val, predictions, average="macro", zero_division=0))
            result = {
                "reg_param": reg_param,
                "status": "PASS",
                "validation_macro_f1": macro_f1,
                "validation_accuracy": float(accuracy_score(y_val, predictions)),
                "warnings": captured_warnings,
            }
            candidate_results.append(result)
            if best is None or macro_f1 > best[0]:
                best = (macro_f1, index, model, result)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            captured_warnings = [str(item.message) for item in caught]
            candidate_results.append({
                "reg_param": reg_param,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": captured_warnings,
            })
    if best is None:
        details = [item.get("error", "unknown fitting failure") for item in candidate_results]
        raise DataContractError(
            "Every QDA reg_param candidate failed. Reduce K, inspect the feature data, "
            f"or revise regularization candidates. Details: {details}"
        )

    run_dir = _new_run_directory(config, root)
    selected_model = best[2]
    model_payload = {
        "model": selected_model,
        "preprocessing": preprocessing,
        "classes": selected_model.classes_.tolist(),
    }
    joblib.dump(model_payload, run_dir / "qda_model.joblib")
    training_id_hashes = (
        bundle.report["train_id_hashes"] + bundle.report["validation_id_hashes"]
    )
    metadata = {
        "config_path": str(resolved_config_path),
        "config": config,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "input_sha256": bundle.report["input_sha256"],
        "feature_modality": config["dataset"].get("feature_modality"),
        "feature_schema": bundle.feature_schema,
        "selected_feature_names": preprocessing["selected_feature_names"],
        "sample_counts": bundle.report["split_counts"],
        "training_and_validation_id_hashes": sorted(training_id_hashes),
        "labels": [str(value) for value in selected_model.classes_.tolist()],
        "random_seed": int(config["random_seed"]),
        "package_versions": _package_versions(),
        "selection": selection_report,
        "reg_param_search": candidate_results,
        "selected_reg_param": float(selected_model.reg_param),
        "validation_selection_metric": "macro_f1",
        "validation_metrics": best[3],
    }
    with (run_dir / "training_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    with (run_dir / "data_validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(bundle.report, stream, indent=2, default=str)
    with (run_dir / "selected_features.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "method": selection_report["selection_method"],
            "k": selection_report["selected_k"],
            "features": selection_report["selected_features"],
        }, stream, indent=2)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qda_static_k50.yaml"),
        help="Path to the QDA YAML configuration.",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = train(args.config)
    except (DataContractError, FileNotFoundError, FileExistsError) as exc:
        print(f"QDA training stopped: {exc}", file=sys.stderr)
        return 2
    print(f"QDA training complete. Frozen model and metadata: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())