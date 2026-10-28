"""Fast contract tests using temporary synthetic datasets only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from groups.group1_qda.src.data import (
    DataContractError,
    load_config,
    load_dataset,
    validate_parquet_dataset,
)
from groups.group1_qda.src.evaluate_qda import evaluate
from groups.group1_qda.src.train_qda import _fit_preprocessing, _new_run_directory, train


def _write_case(
    tmp_path: Path,
    *,
    feature_overrides: dict | None = None,
    manifest_overrides: dict | None = None,
    config_overrides: dict | None = None,
) -> Path:
    dataset_dir = tmp_path / "input"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"sample-{i}" for i in range(36)]
    labels = np.array(["alpha", "beta"] * 18)
    split = np.array(["train"] * 24 + ["val"] * 6 + ["test"] * 6)
    rng = np.random.RandomState(13)
    signal = np.where(labels == "alpha", -2.0, 2.0) + rng.normal(0, 0.15, len(ids))
    features = pd.DataFrame({
        "sample_id": ids,
        "signal": signal,
        "noise": rng.normal(size=len(ids)),
        "constant_train_candidate": np.ones(len(ids)),
        "Class": labels,
        "filename": [f"{item}.apk" for item in ids],
    })
    manifest = pd.DataFrame({
        "sample_id": ids,
        "label": labels,
        "split": split,
    })
    for column, values in (feature_overrides or {}).items():
        features[column] = values
    for column, values in (manifest_overrides or {}).items():
        manifest[column] = values
    feature_path = dataset_dir / "features.csv"
    manifest_path = dataset_dir / "manifest.csv"
    features.to_csv(feature_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    config = {
        "project_root": ".",
        "dataset": {
            "feature_path": str(feature_path.relative_to(tmp_path)),
            "label_source": str(manifest_path.relative_to(tmp_path)),
            "label_column": "label",
            "split_source": str(manifest_path.relative_to(tmp_path)),
            "split_column": "split",
            "sample_id_column": "sample_id",
            "feature_modality": "static",
            "split_values": {"train": "train", "validation": "val", "test": "test"},
            "metadata_columns": [],
            "allowed_labels": ["alpha", "beta"],
        },
        "feature_selection": {"method": "mutual_info", "k": 2},
        "qda": {"reg_param_candidates": [0.1, 0.5, 1.0]},
        "random_seed": 31,
        "output_dir": "outputs/qda_test",
        "run_id": "run-test",
    }
    if config_overrides:
        for key, value in config_overrides.items():
            if key == "dataset":
                config["dataset"].update(value)
            else:
                config[key] = value
    config_path = tmp_path / "qda.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _load(config_path: Path):
    config, _, root = load_config(config_path)
    return config, load_dataset(config, root)


def test_valid_data_contract_excludes_identifiers_and_metadata(tmp_path):
    config_path = _write_case(tmp_path)
    _, bundle = _load(config_path)

    assert bundle.features.index.name == "sample_id"
    assert bundle.features.columns.tolist() == ["signal", "noise", "constant_train_candidate"]
    assert bundle.report["split_counts"] == {"train": 24, "validation": 6, "test": 6}
    assert bundle.report["status"] == "PASS"


@pytest.mark.parametrize("bad_ids", [
    ["duplicate"] * 36,
    [None] + [f"sample-{i}" for i in range(1, 36)],
])
def test_duplicate_or_missing_sample_ids_fail(tmp_path, bad_ids):
    config_path = _write_case(
        tmp_path,
        feature_overrides={"sample_id": bad_ids},
    )
    config, _, root = load_config(config_path)
    with pytest.raises(DataContractError, match="sample ID"):
        load_dataset(config, root)


def test_unmatched_ids_are_not_silently_dropped(tmp_path):
    config_path = _write_case(tmp_path)
    feature_path = tmp_path / "input" / "features.csv"
    features = pd.read_csv(feature_path)
    features.loc[0, "sample_id"] = "unknown-sample"
    features.to_csv(feature_path, index=False)
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="unmatched IDs"):
        load_dataset(config, root)


def test_invalid_labels_and_split_assignments_fail(tmp_path):
    config_path = _write_case(tmp_path, manifest_overrides={"label": ["unknown"] * 36})
    config, _, root = load_config(config_path)
    with pytest.raises(DataContractError, match="allowed_labels"):
        load_dataset(config, root)

    other_path = _write_case(
        tmp_path / "other",
        manifest_overrides={"split": ["mystery"] * 36},
    )
    config, _, root = load_config(other_path)
    with pytest.raises(DataContractError, match="unrecognized assignments"):
        load_dataset(config, root)


def test_duplicate_split_ids_rejected_as_overlapping_assignment(tmp_path):
    config_path = _write_case(tmp_path)
    manifest_path = tmp_path / "input" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[1, "sample_id"] = manifest.loc[0, "sample_id"]
    manifest.to_csv(manifest_path, index=False)
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="duplicate sample IDs"):
        load_dataset(config, root)


def test_non_numeric_feature_is_rejected(tmp_path):
    config_path = _write_case(
        tmp_path,
        feature_overrides={"bad_feature": ["not-a-number"] * 36},
    )
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="Non-numeric values"):
        load_dataset(config, root)


def test_missing_feature_values_are_reported_for_train_only_imputation(tmp_path):
    values = np.where(np.arange(36) == 0, np.nan, np.arange(36, dtype=float))
    config_path = _write_case(tmp_path, feature_overrides={"partially_missing": values})
    config, bundle = _load(config_path)

    assert bundle.report["missing_cells"] == 1
    preprocessing, *_ = _fit_preprocessing(bundle, config)
    assert preprocessing["imputer"].statistics_[bundle.feature_schema.index("partially_missing")] == pytest.approx(
        np.median(values[1:24])
    )


def test_missing_input_file_is_reported(tmp_path):
    config_path = _write_case(
        tmp_path,
        config_overrides={"dataset": {"feature_path": "input/missing.csv"}},
    )
    config, _, root = load_config(config_path)

    with pytest.raises(FileNotFoundError, match="feature_path"):
        load_dataset(config, root)


def test_training_output_cannot_target_group3_real_artifacts(tmp_path):
    config_path = _write_case(
        tmp_path,
        config_overrides={
            "output_dir": "runs",
            "run_id": "real",
        },
    )
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="protected Group 3 artifacts"):
        _new_run_directory(config, root)
    assert not (tmp_path / "runs" / "real").exists()


def test_metadata_and_label_columns_never_become_predictors(tmp_path):
    config_path = _write_case(tmp_path)
    _, bundle = _load(config_path)

    assert "sample_id" not in bundle.features
    assert "Class" not in bundle.features
    assert "filename" not in bundle.features
    assert "label" not in bundle.features
    assert "split" not in bundle.features


def test_opaque_feature_that_copies_labels_is_rejected_as_leakage(tmp_path):
    config_path = _write_case(
        tmp_path,
        feature_overrides={"opaque_copy": [0, 1] * 18},
        manifest_overrides={"label": [0, 1] * 18},
        config_overrides={"dataset": {"allowed_labels": [0, 1]}},
    )
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="leakage"):
        load_dataset(config, root)


def test_duplicate_csv_columns_are_rejected(tmp_path):
    config_path = _write_case(tmp_path)
    feature_path = tmp_path / "input" / "features.csv"
    feature_path.write_text(
        "sample_id,sample_id,signal\nsample-1,sample-1,0.0\n",
        encoding="utf-8",
    )
    config, _, root = load_config(config_path)

    with pytest.raises(DataContractError, match="duplicate column names"):
        load_dataset(config, root)


def _write_parquet_validation_case(
    tmp_path: Path,
    *,
    include_label_leak: bool = False,
) -> tuple[dict, Path]:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = list(range(12))
    labels = [0, 1] * 6
    splits = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
    feature_data = {
        "sample_id": ids,
        "split": splits,
        "signal": [float(value * value + 0.25) for value in range(12)],
        "signal__scaled": [float(value * value + 0.25) / 10 for value in range(12)],
        "constant__scaled": [1.0] * 12,
    }
    if include_label_leak:
        feature_data["opaque_numeric"] = labels
    feature_path = tmp_path / "features.parquet"
    table = pa.Table.from_pydict(feature_data)
    pq.write_table(table, feature_path, row_group_size=4)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ids,
        "label_canonical": labels,
        "split": splits,
    }).to_csv(manifest_path, index=False)
    selection_path = tmp_path / "stage08_feature_selection.json"
    selection_path.write_text(
        json.dumps({
            "static": {"methods": {"mutual_info": {"1": ["signal"]}}}
        }),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "feature_path": str(feature_path),
            "label_source": str(manifest_path),
            "label_column": "label_canonical",
            "split_source": str(manifest_path),
            "split_column": "split",
            "split_values": {"train": "train", "validation": "val", "test": "test"},
            "sample_id_column": "sample_id",
            "feature_modality": "static",
            "allowed_labels": [0, 1],
            "metadata_columns": [],
        },
        "feature_selection": {
            "selection_source": str(selection_path),
            "method": "mutual_info",
            "k": 1,
        },
    }
    return config, tmp_path


def test_parquet_validation_scans_features_in_bounded_batches(tmp_path):
    config, root = _write_parquet_validation_case(tmp_path)

    report = validate_parquet_dataset(config, root, batch_rows=3, batch_columns=1)

    assert report["status"] == "WARNING"
    assert report["parquet_rows"] == 12
    assert report["schema_column_count"] == 5
    assert len(report["feature_schema_sha256"]) == 64
    assert report["sample_ids_unique_and_aligned"] is True
    assert report["split_counts"] == {"train": 6, "validation": 3, "test": 3}
    assert report["configured_selected_features"] == ["signal__scaled"]
    assert report["feature_values_scanned"] == report["scan_expected_values"]
    assert report["scan_batch_rows"] == 3
    assert report["scan_batch_columns"] == 1
    assert report["constant_features"] == ["constant__scaled"]
    assert report["exact_metadata_leakage_columns"] == []


def test_parquet_bounded_validation_rejects_label_leakage(tmp_path):
    config, root = _write_parquet_validation_case(tmp_path, include_label_leak=True)

    with pytest.raises(DataContractError, match="exactly duplicate"):
        validate_parquet_dataset(config, root, batch_rows=3, batch_columns=1)


def test_feature_selection_reproducible_and_fitted_only_on_training_rows(tmp_path):
    config_path = _write_case(tmp_path)
    config, bundle = _load(config_path)
    first = _fit_preprocessing(bundle, config)

    altered = bundle.features.copy()
    altered.loc[bundle.splits != "train", "signal"] = 100000.0
    altered.loc[bundle.splits != "train", "noise"] = -100000.0
    changed_bundle = type(bundle)(
        features=altered,
        labels=bundle.labels,
        splits=bundle.splits,
        report=bundle.report,
        dataset_fingerprint=bundle.dataset_fingerprint,
        feature_schema=bundle.feature_schema,
        id_column=bundle.id_column,
    )
    second = _fit_preprocessing(changed_bundle, config)

    assert first[0]["selected_feature_names"] == second[0]["selected_feature_names"]
    # The selector and scaler saw only train rows; the extreme held-out changes do not alter them.
    np.testing.assert_allclose(first[0]["scaler"].mean_, second[0]["scaler"].mean_)


def test_train_freezes_validation_choice_and_evaluation_uses_test_partition(tmp_path):
    config_path = _write_case(tmp_path)
    run_dir = train(config_path)
    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))

    assert metadata["validation_selection_metric"] == "macro_f1"
    assert "validation_metrics" in metadata
    assert not any("test_" in key for key in metadata["validation_metrics"])
    result_dir = evaluate(config_path, run_dir)
    results = json.loads((result_dir / "evaluation_results.json").read_text(encoding="utf-8"))
    assert results["split"] == "test"
    assert results["metrics"]["test_sample_count"] == 6
    assert {
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "per_class_metrics", "confusion_matrix",
    } <= set(results["metrics"])


def test_changed_feature_schema_requires_and_rejects_incompatible_evaluation(tmp_path):
    config_path = _write_case(tmp_path)
    run_dir = train(config_path)
    changed_config = _write_case(
        tmp_path / "changed",
        feature_overrides={"new_feature": np.arange(36)},
    )
    with pytest.raises(DataContractError, match="fingerprint differs"):
        evaluate(changed_config, run_dir)
    with pytest.raises(DataContractError, match="schema is incompatible"):
        evaluate(changed_config, run_dir, allow_compatible_dataset_change=True)