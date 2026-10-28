"""Evaluate a frozen Group 1 QDA model on its configured official test split only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from groups.group1_qda.src.data import (
    DataContractError,
    _id_hash,
    load_config,
    load_dataset,
)
from groups.group1_qda.src.train_qda import _transform


def evaluate(
    config_path: str | Path,
    model_dir: str | Path,
    *,
    allow_compatible_dataset_change: bool = False,
) -> Path:
    config, resolved_config_path, root = load_config(config_path)
    model_path = Path(model_dir).expanduser().resolve() / "qda_model.joblib"
    metadata_path = Path(model_dir).expanduser().resolve() / "training_metadata.json"
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen QDA artifact is missing: {path}")
    payload = joblib.load(model_path)
    with metadata_path.open(encoding="utf-8") as stream:
        training_metadata = json.load(stream)

    bundle = load_dataset(config, root)
    same_dataset = bundle.dataset_fingerprint == training_metadata["dataset_fingerprint"]
    if not same_dataset and not allow_compatible_dataset_change:
        raise DataContractError(
            "Dataset fingerprint differs from training. Evaluation stopped. For a "
            "deliberate compatible external evaluation, pass "
            "--allow-compatible-dataset-change; input feature schema, labels, split "
            "contract, and train/validation ID disjointness will still be checked."
        )

    expected_schema = training_metadata["feature_schema"]
    if set(bundle.feature_schema) != set(expected_schema) or len(bundle.feature_schema) != len(expected_schema):
        missing = sorted(set(expected_schema) - set(bundle.feature_schema))
        extra = sorted(set(bundle.feature_schema) - set(expected_schema))
        raise DataContractError(
            f"Evaluation feature schema is incompatible. Missing columns: {missing[:10]}; "
            f"unexpected columns: {extra[:10]}."
        )
    labels = {str(value) for value in bundle.labels.unique()}
    expected_labels = set(training_metadata["labels"])
    if labels != expected_labels:
        raise DataContractError(
            f"Evaluation label set differs from training. Expected {sorted(expected_labels)}, "
            f"found {sorted(labels)}."
        )
    training_modality = training_metadata.get("feature_modality")
    evaluation_modality = config["dataset"].get("feature_modality")
    if training_modality != evaluation_modality:
        raise DataContractError(
            f"Evaluation modality {evaluation_modality!r} differs from model modality "
            f"{training_modality!r}."
        )

    test_mask = bundle.splits == "test"
    if not test_mask.any():
        raise DataContractError("No samples are assigned to the configured official test split.")
    if not same_dataset:
        train_val_hashes = set(training_metadata["training_and_validation_id_hashes"])
        overlapping = [
            value for value in bundle.features.index[test_mask]
            if _id_hash(value) in train_val_hashes
        ]
        if overlapping:
            raise DataContractError(
                f"Compatible-dataset evaluation has {len(overlapping)} test IDs that overlap "
                "the training or validation IDs."
            )
    X_test_frame = bundle.features.loc[test_mask]
    y_test = bundle.labels.loc[test_mask].to_numpy()
    X_test = _transform(X_test_frame, payload["preprocessing"])
    predictions = payload["model"].predict(X_test)
    class_labels = payload["classes"]
    report = classification_report(
        y_test, predictions, labels=class_labels, output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "per_class_metrics": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=class_labels).tolist(),
        "confusion_matrix_labels": [str(value) for value in class_labels],
        "test_sample_count": int(len(y_test)),
    }
    evaluation_dir = Path(model_dir).expanduser().resolve() / "evaluations"
    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    evaluation_dir = evaluation_dir / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "evaluation_config": config,
        "evaluation_config_path": str(resolved_config_path),
        "training_dataset_fingerprint": training_metadata["dataset_fingerprint"],
        "evaluation_dataset_fingerprint": bundle.dataset_fingerprint,
        "input_fingerprint_differences": {
            key: {
                "training": training_metadata["input_sha256"].get(key),
                "evaluation": bundle.report["input_sha256"].get(key),
            }
            for key in sorted(
                set(training_metadata["input_sha256"]) | set(bundle.report["input_sha256"])
            )
            if training_metadata["input_sha256"].get(key)
            != bundle.report["input_sha256"].get(key)
        },
        "dataset_fingerprint_matches_training": same_dataset,
        "compatible_dataset_change_explicitly_allowed": bool(
            allow_compatible_dataset_change and not same_dataset
        ),
        "feature_schema_matches_training": True,
        "feature_modality": config["dataset"].get("feature_modality"),
        "split": "test",
        "metrics": metrics,
        "evaluation_data_validation_report": bundle.report,
    }
    with (evaluation_dir / "evaluation_results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, default=str)
    return evaluation_dir


def main(argv: list[str] | None = None) -> int:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "qda_static_k50.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config), help="QDA YAML configuration.")
    parser.add_argument("--model-dir", required=True, help="Run directory containing qda_model.joblib.")
    parser.add_argument(
        "--allow-compatible-dataset-change",
        action="store_true",
        help="Explicitly permit changed input contents after strict schema/label/ID checks.",
    )
    args = parser.parse_args(argv)
    try:
        evaluation_dir = evaluate(
            args.config,
            args.model_dir,
            allow_compatible_dataset_change=args.allow_compatible_dataset_change,
        )
    except (DataContractError, FileNotFoundError) as exc:
        print(f"QDA evaluation stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Test evaluation complete. Results: {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())