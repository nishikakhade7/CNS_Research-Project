"""Config-driven dataset loading and strict QDA input-contract validation."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_ALIASES = {
    "train": {"train"},
    "validation": {"validation", "valid", "val"},
    "test": {"test"},
}
_METADATA_NAMES = {
    "id", "sampleid", "appid", "filename", "file", "filepath", "path",
    "label", "labels", "target", "class", "split", "fold", "sha", "sha1",
    "sha256", "md5", "hash", "package", "packagename", "groundtruth",
    "malware", "ismalware",
}
_METADATA_PATTERNS = (
    re.compile(r"(^|_)(id|label|target|split|fold)(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)(filename|filepath|sha256|sha1|md5|hash)(_|$)", re.IGNORECASE),
)


class DataContractError(ValueError):
    """Raised when input data cannot be joined or modeled without guessing."""


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series
    splits: pd.Series
    report: dict[str, Any]
    dataset_fingerprint: str
    feature_schema: list[str]
    id_column: str


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    """Load YAML and resolve all configured paths relative to project_root."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise DataContractError(f"Invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DataContractError(f"Configuration must be a YAML mapping: {path}")

    required = ("dataset", "feature_selection", "qda", "random_seed", "output_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise DataContractError(f"Configuration is missing required sections: {missing}")

    root_setting = config.get("project_root", ".")
    root = (path.parent / root_setting).resolve()
    return config, path, root


def resolve_config_path(value: str | Path, root: Path, *, field: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configured {field} does not exist: {resolved}")
    return resolved


def _read_table(path: Path, *, field: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            try:
                header = next(csv.reader(stream))
            except StopIteration as exc:
                raise DataContractError(f"{field} file is empty: {path}") from exc
        duplicated = sorted(name for name, count in Counter(header).items() if count > 1)
        if duplicated:
            raise DataContractError(f"{field} CSV has duplicate column names: {duplicated}")
        frame = pd.read_csv(path)
    else:
        raise DataContractError(
            f"Unsupported {field} file type {suffix!r} at {path}; use CSV or Parquet."
        )
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise DataContractError(f"{field} has duplicate column names: {duplicated}")
    non_string_columns = [column for column in frame.columns if not isinstance(column, str)]
    if non_string_columns:
        raise DataContractError(
            f"{field} has non-string column names: {non_string_columns[:10]}. "
            "Rename columns to strings explicitly before loading."
        )
    if not len(frame.columns):
        raise DataContractError(f"{field} has no columns: {path}")
    return frame


def _require_column(frame: pd.DataFrame, column: str, *, field: str, path: Path) -> None:
    if column not in frame.columns:
        raise DataContractError(
            f"{field} {path} has no configured column {column!r}. "
            f"Available columns: {frame.columns.tolist()[:30]}"
        )


def _validate_ids(frame: pd.DataFrame, column: str, *, field: str, path: Path) -> None:
    _require_column(frame, column, field=field, path=path)
    values = frame[column]
    if values.isna().any():
        raise DataContractError(
            f"{field} {path} has {int(values.isna().sum())} missing values in sample ID "
            f"column {column!r}."
        )
    duplicates = values[values.duplicated(keep=False)]
    if not duplicates.empty:
        examples = duplicates.astype(str).drop_duplicates().head(5).tolist()
        raise DataContractError(
            f"{field} {path} has duplicate sample IDs in {column!r} "
            f"(examples: {examples}). Expected one row per sample."
        )


def _feature_is_metadata(
    name: str,
    *,
    id_column: str,
    label_column: str,
    split_column: str,
    configured_metadata: set[str],
) -> bool:
    if name in configured_metadata or name in {id_column, label_column, split_column}:
        return True
    lowered = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    compact = normalized.replace("_", "")
    if lowered.endswith("__scaled"):
        return True
    if normalized in _METADATA_NAMES or compact in _METADATA_NAMES:
        return True
    if any(pattern.search(normalized) for pattern in _METADATA_PATTERNS):
        return True
    return False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    left_values = left.reset_index(drop=True)
    right_values = right.reset_index(drop=True)
    try:
        left_numeric = pd.to_numeric(left_values, errors="raise").to_numpy(dtype=float)
        right_numeric = pd.to_numeric(right_values, errors="raise").to_numpy(dtype=float)
        return bool(np.array_equal(left_numeric, right_numeric, equal_nan=True))
    except (TypeError, ValueError):
        return left_values.astype("string").equals(right_values.astype("string"))


def load_dataset(config: dict[str, Any], project_root: Path) -> DatasetBundle:
    """Load feature, label, and official split sources and enforce exact ID joins."""
    ds = config["dataset"]
    id_column = str(ds.get("sample_id_column", "")).strip()
    label_column = str(ds.get("label_column", "")).strip()
    split_column = str(ds.get("split_column", "")).strip()
    for key, value in (
        ("sample_id_column", id_column),
        ("label_column", label_column),
        ("split_column", split_column),
    ):
        if not value:
            raise DataContractError(f"dataset.{key} must be configured explicitly.")
    modality = ds.get("feature_modality")
    if modality not in {"static", "fused"}:
        raise DataContractError(
            "dataset.feature_modality must explicitly be either 'static' or 'fused'."
        )

    feature_path = resolve_config_path(ds.get("feature_path", ""), project_root, field="feature_path")
    label_path = resolve_config_path(ds.get("label_source", ""), project_root, field="label_source")
    split_path = resolve_config_path(ds.get("split_source", ""), project_root, field="split_source")
    feature_table = _read_table(feature_path, field="feature dataset")
    label_table = _read_table(label_path, field="label source")
    split_table = _read_table(split_path, field="split source")

    for frame, path, field in (
        (feature_table, feature_path, "feature dataset"),
        (label_table, label_path, "label source"),
        (split_table, split_path, "split source"),
    ):
        _validate_ids(frame, id_column, field=field, path=path)
    _require_column(label_table, label_column, field="label source", path=label_path)
    _require_column(split_table, split_column, field="split source", path=split_path)

    labels = label_table[[id_column, label_column]].copy()
    splits = split_table[[id_column, split_column]].copy()
    labels.rename(columns={label_column: "__qda_label"}, inplace=True)
    splits.rename(columns={split_column: "__qda_split"}, inplace=True)
    if not (
        feature_table[id_column].dtype == label_table[id_column].dtype
        and feature_table[id_column].dtype == split_table[id_column].dtype
    ):
        raise DataContractError(
            "Sample ID column types differ across feature, label, and split sources. "
            "Normalize them explicitly in the input files; automatic coercion could "
            "create incorrect sample matches."
        )
    try:
        joined = feature_table[[id_column]].merge(
            labels, on=id_column, how="outer", validate="one_to_one", indicator="__label_join"
        )
        joined = joined.merge(
            splits, on=id_column, how="outer", validate="one_to_one", indicator="__split_join"
        )
    except (pd.errors.MergeError, ValueError) as exc:
        raise DataContractError(f"Input sources do not have one-to-one sample IDs: {exc}") from exc
    for indicator, field, path in (
        ("__label_join", "label source", label_path),
        ("__split_join", "split source", split_path),
    ):
        unmatched = joined.loc[joined[indicator] != "both", id_column]
        if not unmatched.empty:
            examples = unmatched.astype(str).head(5).tolist()
            raise DataContractError(
                f"Sample IDs do not match exactly between the feature dataset and {field} "
                f"{path}; {len(unmatched)} unmatched IDs (examples: {examples}). "
                "No samples were discarded."
            )
    if joined["__qda_label"].isna().any():
        raise DataContractError("Label source contains missing/invalid labels.")
    label_values = joined["__qda_label"]
    if label_values.map(
        lambda value: isinstance(value, str) and not value.strip()
    ).any():
        raise DataContractError("Label source contains empty-string labels.")
    if pd.api.types.is_numeric_dtype(label_values):
        finite = np.isfinite(label_values.to_numpy(dtype=float))
        if not finite.all():
            raise DataContractError("Label source contains infinite numeric labels.")
    allowed_labels = ds.get("allowed_labels")
    if allowed_labels is not None:
        if not isinstance(allowed_labels, list) or not allowed_labels:
            raise DataContractError(
                "dataset.allowed_labels must be null or a non-empty list."
            )
        unexpected_labels = sorted(
            set(label_values.unique()) - set(allowed_labels), key=str
        )
        if unexpected_labels:
            raise DataContractError(
                f"Label source contains values outside configured allowed_labels: "
                f"{unexpected_labels[:20]}"
            )
    if joined["__qda_split"].isna().any():
        raise DataContractError("Split source contains missing split assignments.")

    raw_split_values = joined["__qda_split"]
    configured_splits = ds.get("split_values", {})
    if configured_splits:
        if set(configured_splits) != set(SPLIT_NAMES):
            raise DataContractError(
                "dataset.split_values must map exactly train, validation, and test."
            )
        split_map = {value: name for name, value in configured_splits.items()}
        if len(split_map) != len(configured_splits):
            raise DataContractError(
                "dataset.split_values must map train, validation, and test to distinct values."
            )
        unknown = sorted(set(raw_split_values.unique()) - set(split_map))
        if unknown:
            raise DataContractError(f"Split source has unrecognized assignments: {unknown}")
        normalized_splits = raw_split_values.map(split_map)
    else:
        normalized_splits = raw_split_values.map(
            lambda value: next(
                (name for name, aliases in SPLIT_ALIASES.items() if str(value).lower() in aliases),
                None,
            )
        )
        unknown = sorted(raw_split_values[normalized_splits.isna()].astype(str).unique().tolist())
        if unknown:
            raise DataContractError(
                f"Cannot safely identify split assignment(s) {unknown}; configure "
                "dataset.split_values explicitly."
            )
    if set(normalized_splits.unique()) != set(SPLIT_NAMES):
        absent = sorted(set(SPLIT_NAMES) - set(normalized_splits.unique()))
        raise DataContractError(f"Official split source has empty partition(s): {absent}.")

    metadata_setting = ds.get("metadata_columns", [])
    if not isinstance(metadata_setting, list) or any(
        not isinstance(name, str) for name in metadata_setting
    ):
        raise DataContractError("dataset.metadata_columns must be a list of column names.")
    configured_metadata = set(metadata_setting)
    feature_columns = [
        column for column in feature_table.columns
        if not _feature_is_metadata(
            str(column),
            id_column=id_column,
            label_column=label_column,
            split_column=split_column,
            configured_metadata=configured_metadata,
        )
    ]
    if not feature_columns:
        raise DataContractError("No predictive feature columns remain after metadata exclusion.")

    feature_frame = feature_table.set_index(id_column)[feature_columns].copy()
    feature_frame.index.name = id_column
    non_numeric: dict[str, list[str]] = {}
    for column in feature_frame:
        converted = pd.to_numeric(feature_frame[column], errors="coerce")
        invalid = feature_frame[column].notna() & converted.isna()
        if invalid.any():
            non_numeric[column] = feature_frame.loc[invalid, column].astype(str).unique()[:3].tolist()
        else:
            feature_frame[column] = converted.astype(float)
    if non_numeric:
        details = {key: values for key, values in list(non_numeric.items())[:8]}
        raise DataContractError(
            f"Predictive features must be numeric or numerically parseable. "
            f"Non-numeric values found: {details}"
        )
    infinities = np.isinf(feature_frame.to_numpy(dtype=float))
    if infinities.any():
        row, col = np.argwhere(infinities)[0]
        raise DataContractError(
            f"Infinite feature value in sample {feature_frame.index[row]!r}, "
            f"column {feature_frame.columns[col]!r}."
        )
    missing_by_feature = feature_frame.isna().sum()
    all_missing = missing_by_feature[missing_by_feature == len(feature_frame)].index.tolist()
    if all_missing:
        raise DataContractError(
            f"Feature(s) contain no observed values and cannot be imputed: {all_missing[:10]}"
        )

    indexed_join = joined.set_index(id_column)
    label_series = indexed_join["__qda_label"].reindex(feature_frame.index)
    split_series = normalized_splits.set_axis(joined[id_column]).reindex(feature_frame.index)
    if label_series.isna().any() or split_series.isna().any():
        raise DataContractError("Internal alignment failure after sample-ID joins.")
    feature_unique_counts = feature_frame.nunique(dropna=False)
    metadata_references = {
        id_column: pd.Series(feature_frame.index, index=feature_frame.index),
        label_column: label_series,
        split_column: split_series,
    }
    reference_unique_counts = {
        name: values.nunique(dropna=False) for name, values in metadata_references.items()
    }
    leaked_columns = [
        feature_name
        for feature_name in feature_frame.columns
        for metadata_name, values in metadata_references.items()
        if feature_unique_counts[feature_name] == reference_unique_counts[metadata_name]
        and _same_values(feature_frame[feature_name], values)
    ]
    if leaked_columns:
        raise DataContractError(
            "Potential target/identity/split leakage: predictive column(s) exactly "
            f"duplicate a configured metadata source: {sorted(set(leaked_columns))[:20]}. "
            "Exclude them using dataset.metadata_columns or correct the feature source."
        )
    partition_ids = {
        name: set(feature_frame.index[split_series == name].tolist()) for name in SPLIT_NAMES
    }
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap = partition_ids[left] & partition_ids[right]
            if overlap:
                raise DataContractError(
                    f"Official partitions overlap: {left}/{right} share {len(overlap)} IDs."
                )

    constant_columns = feature_frame.nunique(dropna=True)
    constant_features = constant_columns[constant_columns <= 1].index.tolist()
    feature_frame.index.name = id_column
    label_series.name = label_column
    split_series.name = "split"
    fingerprint_inputs = {
        "feature_sha256": _file_hash(feature_path),
        "label_sha256": _file_hash(label_path),
        "split_sha256": _file_hash(split_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "status": "PASS",
        "feature_path": str(feature_path),
        "label_source": str(label_path),
        "split_source": str(split_path),
        "feature_modality": ds.get("feature_modality"),
        "sample_count": int(len(feature_frame)),
        "feature_count_before_variance_filter": int(len(feature_columns)),
        "missing_cells": int(feature_frame.isna().sum().sum()),
        "missing_values_by_feature": {
            str(key): int(value) for key, value in missing_by_feature.items() if value
        },
        "constant_features_detected": constant_features,
        "split_counts": {str(k): int(v) for k, v in split_series.value_counts().items()},
        "label_counts": {
            str(label): int(count) for label, count in label_series.value_counts().items()
        },
        "excluded_metadata_columns": [
            str(column) for column in feature_table.columns if column not in feature_columns
        ],
        "feature_schema": [str(column) for column in feature_columns],
        "input_sha256": fingerprint_inputs,
        "dataset_fingerprint": fingerprint,
        "train_id_hashes": sorted(_id_hash(value) for value in partition_ids["train"]),
        "validation_id_hashes": sorted(_id_hash(value) for value in partition_ids["validation"]),
        "test_id_hashes": sorted(_id_hash(value) for value in partition_ids["test"]),
    }
    return DatasetBundle(
        features=feature_frame,
        labels=label_series,
        splits=split_series,
        report=report,
        dataset_fingerprint=fingerprint,
        feature_schema=[str(column) for column in feature_columns],
        id_column=id_column,
    )


def _load_selection_columns(
    config: dict[str, Any],
    feature_path: Path,
    project_root: Path,
    schema_names: set[str],
) -> tuple[Path, list[str]]:
    """Resolve configured ranking names to scaled columns without reading features."""
    selection_config = config["feature_selection"]
    ds = config["dataset"]
    configured_path = selection_config.get("selection_source")
    selection_path = (
        Path(configured_path).expanduser()
        if configured_path
        else feature_path.parent / "stage08_feature_selection.json"
    )
    if not selection_path.is_absolute():
        selection_path = (
            (project_root / selection_path).resolve()
            if configured_path
            else selection_path.resolve()
        )
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Feature selection report is required for configured modality/method/K: "
            f"{selection_path}. Set feature_selection.selection_source explicitly."
        )
    try:
        with selection_path.open(encoding="utf-8") as stream:
            selection = json.load(stream)
        modality = ds["feature_modality"]
        method = selection_config["method"]
        k = str(selection_config["k"])
        base_names = selection[modality]["methods"][method][k]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            f"Cannot find feature list for modality={ds.get('feature_modality')!r}, "
            f"method={selection_config.get('method')!r}, K={selection_config.get('k')!r} "
            f"in {selection_path}."
        ) from exc
    if not isinstance(base_names, list) or not base_names:
        raise DataContractError(
            f"Configured feature-selection entry is not a non-empty list: {selection_path}"
        )
    if len(base_names) != len(set(base_names)):
        raise DataContractError(f"Feature selection report has duplicate names: {selection_path}")
    selected_columns = [f"{name}__scaled" for name in base_names]
    missing = [name for name in selected_columns if name not in schema_names]
    if missing:
        raise DataContractError(
            f"Configured scaled feature columns are missing from {feature_path}: {missing[:10]}"
        )
    return selection_path.resolve(), selected_columns


def validate_parquet_dataset(
    config: dict[str, Any],
    project_root: Path,
    *,
    batch_rows: int = 256,
    batch_columns: int = 256,
) -> dict[str, Any]:
    """Validate a Parquet dataset in bounded batches, without materializing its matrix.

    Feature values are read in column groups and row batches. The only in-memory table
    is the comparatively small official label/split manifest.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_rows < 1 or batch_columns < 1:
        raise ValueError("batch_rows and batch_columns must be positive.")
    ds = config["dataset"]
    id_column = str(ds.get("sample_id_column", "")).strip()
    label_column = str(ds.get("label_column", "")).strip()
    split_column = str(ds.get("split_column", "")).strip()
    if not id_column or not label_column or not split_column:
        raise DataContractError(
            "sample_id_column, label_column, and split_column must be configured explicitly."
        )
    modality = ds.get("feature_modality")
    if modality not in {"static", "fused"}:
        raise DataContractError("dataset.feature_modality must be 'static' or 'fused'.")

    feature_path = resolve_config_path(ds.get("feature_path", ""), project_root, field="feature_path")
    if feature_path.suffix.lower() not in {".parquet", ".pq"}:
        raise DataContractError(
            "Memory-efficient data validation currently requires a Parquet feature dataset."
        )
    label_path = resolve_config_path(ds.get("label_source", ""), project_root, field="label_source")
    split_path = resolve_config_path(ds.get("split_source", ""), project_root, field="split_source")
    try:
        parquet = pq.ParquetFile(feature_path, memory_map=False)
        metadata = parquet.metadata
        schema = parquet.schema_arrow
    except (pa.ArrowException, OSError) as exc:
        raise DataContractError(f"Cannot read Parquet metadata for {feature_path}: {exc}") from exc
    names = schema.names
    if len(names) != len(set(names)):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise DataContractError(f"Feature Parquet has duplicate column names: {duplicates[:20]}")
    if id_column not in names:
        raise DataContractError(f"Feature Parquet has no configured ID column {id_column!r}.")
    id_type = schema.field(id_column).type
    if not (pa.types.is_integer(id_type) or pa.types.is_string(id_type) or pa.types.is_large_string(id_type)):
        raise DataContractError(
            f"Feature ID column {id_column!r} has unsupported type {id_type}; use integer or string IDs."
        )

    labels = _read_table(label_path, field="label source")
    splits = labels if split_path == label_path else _read_table(split_path, field="split source")
    _validate_ids(labels, id_column, field="label source", path=label_path)
    _validate_ids(splits, id_column, field="split source", path=split_path)
    _require_column(labels, label_column, field="label source", path=label_path)
    _require_column(splits, split_column, field="split source", path=split_path)

    manifest_ids = labels[id_column].tolist()
    manifest_set = set(manifest_ids)
    if len(manifest_set) != len(manifest_ids):
        raise DataContractError("Official label source contains duplicate sample IDs.")
    label_id_type = pa.array(labels[id_column]).type
    compatible_id_types = (
        id_type == label_id_type
        or (
            pa.types.is_string(id_type) or pa.types.is_large_string(id_type)
        ) and (
            pa.types.is_string(label_id_type) or pa.types.is_large_string(label_id_type)
        )
    )
    if not compatible_id_types:
        raise DataContractError(
            f"Feature ID type {id_type} differs from configured label-source ID type "
            f"{labels[id_column].dtype}; normalize them explicitly."
        )
    label_by_id = dict(zip(labels[id_column].tolist(), labels[label_column].tolist()))
    split_by_id = dict(zip(splits[id_column].tolist(), splits[split_column].tolist()))
    if set(label_by_id) != set(split_by_id):
        raise DataContractError("Label and official split sources have different sample ID sets.")
    if any(pd.isna(value) for value in label_by_id.values()):
        raise DataContractError("Official manifest contains missing labels.")
    if any(isinstance(value, str) and not value.strip() for value in label_by_id.values()):
        raise DataContractError("Official manifest contains empty-string labels.")
    if pd.api.types.is_numeric_dtype(labels[label_column]):
        if not np.isfinite(labels[label_column].to_numpy(dtype=float)).all():
            raise DataContractError("Official manifest contains infinite numeric labels.")
    allowed_labels = ds.get("allowed_labels")
    observed_labels = set(label_by_id.values())
    if allowed_labels is not None:
        if not isinstance(allowed_labels, list) or not allowed_labels:
            raise DataContractError("dataset.allowed_labels must be null or a non-empty list.")
        unexpected_labels = observed_labels - set(allowed_labels)
        if unexpected_labels:
            raise DataContractError(
                f"Manifest has labels outside dataset.allowed_labels: "
                f"{sorted(unexpected_labels, key=str)[:20]}"
            )
    if any(pd.isna(value) for value in split_by_id.values()):
        raise DataContractError("Official manifest contains missing split assignments.")

    configured_splits = ds.get("split_values", {})
    if configured_splits:
        if set(configured_splits) != set(SPLIT_NAMES) or len(set(configured_splits.values())) != 3:
            raise DataContractError(
                "dataset.split_values must map train, validation, and test to distinct values."
            )
        split_map = {value: name for name, value in configured_splits.items()}
        unknown = set(split_by_id.values()) - set(split_map)
        if unknown:
            raise DataContractError(
                f"Official manifest has unrecognized split values: {sorted(unknown, key=str)}"
            )
        normalized_split_by_id = {key: split_map[value] for key, value in split_by_id.items()}
    else:
        normalized_split_by_id = {}
        for key, value in split_by_id.items():
            normalized = next(
                (
                    name for name, aliases in SPLIT_ALIASES.items()
                    if str(value).lower() in aliases
                ),
                None,
            )
            if normalized is None:
                raise DataContractError(
                    f"Cannot identify official split value {value!r}; configure split_values."
                )
            normalized_split_by_id[key] = normalized
    if set(normalized_split_by_id.values()) != set(SPLIT_NAMES):
        raise DataContractError("Official manifest must contain train, validation, and test.")

    configured_metadata = set(ds.get("metadata_columns", []))
    excluded_names = {
        name for name in names
        if _feature_is_metadata(
            name[:-len("__scaled")] if name.endswith("__scaled") else name,
            id_column=id_column,
            label_column=label_column,
            split_column=split_column,
            configured_metadata=configured_metadata,
        )
    }
    feature_columns = [name for name in names if name not in excluded_names]
    if not feature_columns:
        raise DataContractError("No feature columns remain after metadata exclusion.")
    non_numeric_columns = [
        name for name in feature_columns
        if not (
            pa.types.is_integer(schema.field(name).type)
            or pa.types.is_floating(schema.field(name).type)
            or pa.types.is_decimal(schema.field(name).type)
        )
    ]
    if non_numeric_columns:
        raise DataContractError(
            f"Feature schema has non-numeric predictive columns: {non_numeric_columns[:20]}"
        )
    selection_path, selected_columns = _load_selection_columns(
        config, feature_path, project_root, set(names)
    )
    if any(name not in feature_columns and name != id_column for name in selected_columns):
        raise DataContractError("Selected feature list contains excluded metadata columns.")

    try:
        id_table = parquet.read(columns=[id_column], use_threads=False)
    except (pa.ArrowException, OSError) as exc:
        raise DataContractError(f"Cannot read projected IDs from {feature_path}: {exc}") from exc
    feature_ids = id_table.column(id_column).to_pylist()
    if any(value is None for value in feature_ids):
        raise DataContractError(f"Feature Parquet has missing IDs in {id_column!r}.")
    if len(feature_ids) != len(set(feature_ids)):
        raise DataContractError(f"Feature Parquet has duplicate IDs in {id_column!r}.")
    feature_id_set = set(feature_ids)
    if feature_id_set != manifest_set:
        raise DataContractError(
            f"Feature and official manifest IDs differ: feature-only={len(feature_id_set-manifest_set)}, "
            f"manifest-only={len(manifest_set-feature_id_set)}."
        )
    del id_table, feature_ids, feature_id_set

    value_columns = [name for name in feature_columns if name != id_column]
    constant_columns: list[str] = []
    columns_with_missing: list[str] = []
    columns_with_nonfinite: list[str] = []
    leakage_columns: set[str] = set()
    observed_value_counts: dict[str, int] = {}
    total_rows_scanned = 0
    batch_count = 0
    for start in range(0, len(value_columns), batch_columns):
        column_group = value_columns[start:start + batch_columns]
        read_columns = [id_column, *column_group]
        group_missing = np.zeros(len(column_group), dtype=bool)
        group_nonfinite = np.zeros(len(column_group), dtype=bool)
        group_min = np.full(len(column_group), np.inf)
        group_max = np.full(len(column_group), -np.inf)
        group_leakage = np.ones((3, len(column_group)), dtype=bool)
        group_comparable = np.zeros(3, dtype=bool)
        group_observed_count = np.zeros(len(column_group), dtype=np.int64)
        try:
            batches = parquet.iter_batches(
                batch_size=batch_rows,
                columns=read_columns,
                use_threads=False,
            )
            for batch in batches:
                batch_count += 1
                total_rows_scanned += batch.num_rows * len(column_group)
                batch_ids = batch.column(0).to_pylist()
                if len(batch_ids) != batch.num_rows or any(item not in manifest_set for item in batch_ids):
                    raise DataContractError(
                        "Feature batch IDs do not align to the official manifest."
                    )
                batch_labels = [label_by_id[item] for item in batch_ids]
                batch_splits = [normalized_split_by_id[item] for item in batch_ids]
                matrix = np.column_stack([
                    np.asarray(
                        batch.column(column_index).to_numpy(zero_copy_only=False),
                        dtype=np.float64,
                    )
                    for column_index in range(1, len(column_group) + 1)
                ])
                nulls = np.fromiter(
                    (batch.column(i).null_count > 0 for i in range(1, len(column_group) + 1)),
                    dtype=bool,
                    count=len(column_group),
                )
                group_missing |= nulls
                finite = np.isfinite(matrix)
                group_nonfinite |= ~finite.all(axis=0)
                group_observed_count += finite.sum(axis=0)
                finite_min = np.min(np.where(finite, matrix, np.inf), axis=0)
                finite_max = np.max(np.where(finite, matrix, -np.inf), axis=0)
                group_min = np.minimum(group_min, finite_min)
                group_max = np.maximum(group_max, finite_max)

                for source_index, raw_values in enumerate(
                    (batch_ids, batch_labels, batch_splits)
                ):
                    try:
                        target_values = np.asarray(raw_values, dtype=np.float64)
                    except (TypeError, ValueError):
                        group_comparable[source_index] = False
                        continue
                    if np.isfinite(target_values).all():
                        group_comparable[source_index] = True
                        group_leakage[source_index] &= np.all(
                            matrix == target_values[:, np.newaxis], axis=0
                        )
        except DataContractError:
            raise
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise DataContractError(
                f"Failed scanning feature column group beginning {column_group[0]!r}: {exc}"
            ) from exc
        columns_with_missing.extend(
            name for name, has_missing in zip(column_group, group_missing) if has_missing
        )
        columns_with_nonfinite.extend(
            name for name, has_nonfinite in zip(column_group, group_nonfinite) if has_nonfinite
        )
        constant_columns.extend(
            name for name, low, high in zip(column_group, group_min, group_max)
            if low == high
        )
        observed_value_counts.update(
            zip(column_group, (int(value) for value in group_observed_count))
        )
        leakage_columns.update(
            name for index, name in enumerate(column_group)
            if (group_leakage[:, index] & group_comparable).any()
        )
    if total_rows_scanned != len(value_columns) * metadata.num_rows:
        # Iteration scans each column group independently; each group must cover all rows.
        expected = len(value_columns) * metadata.num_rows
        raise DataContractError(
            f"Incomplete bounded feature scan: scanned {total_rows_scanned} values across "
            f"column groups, expected {expected}."
        )
    if leakage_columns:
        raise DataContractError(
            f"Predictive feature(s) exactly duplicate sample IDs, labels, or split "
            f"assignments: {sorted(leakage_columns)[:20]}"
        )
    all_missing_columns = [
        name for name, count in observed_value_counts.items() if count == 0
    ]
    if all_missing_columns:
        raise DataContractError(
            "Feature(s) contain no finite observed values: "
            f"{all_missing_columns[:20]}"
        )

    schema_description = [
        (name, str(schema.field(name).type)) for name in names
    ]
    schema_digest = hashlib.sha256(
        json.dumps(schema_description, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    report = {
        "status": "PASS",
        "feature_path": str(feature_path),
        "label_source": str(label_path),
        "split_source": str(split_path),
        "selection_source": str(selection_path),
        "feature_modality": modality,
        "parquet_rows": int(metadata.num_rows),
        "parquet_row_groups": int(metadata.num_row_groups),
        "schema_column_count": int(metadata.num_columns),
        "feature_column_count": int(len(feature_columns)),
        "feature_schema_sha256": schema_digest,
        "feature_schema_first_columns": schema.names[:10],
        "feature_schema_last_columns": schema.names[-10:],
        "sample_id_count": len(manifest_set),
        "sample_ids_unique_and_aligned": True,
        "label_counts": {
            str(key): int(value) for key, value in Counter(label_by_id.values()).items()
        },
        "split_counts": {
            name: int(sum(value == name for value in normalized_split_by_id.values()))
            for name in SPLIT_NAMES
        },
        "configured_selected_features": selected_columns,
        "selected_feature_count": len(selected_columns),
        "feature_values_scanned": int(total_rows_scanned),
        "scan_expected_values": int(len(value_columns) * metadata.num_rows),
        "scan_batch_rows": batch_rows,
        "scan_batch_columns": batch_columns,
        "scan_batch_count": batch_count,
        "columns_with_missing_values": sorted(set(columns_with_missing)),
        "columns_with_nonfinite_values": sorted(set(columns_with_nonfinite)),
        "constant_features": constant_columns,
        "exact_metadata_leakage_columns": sorted(leakage_columns),
        "all_feature_values_scanned": total_rows_scanned == len(value_columns) * metadata.num_rows,
    }
    if columns_with_missing or constant_columns:
        report["status"] = "WARNING"
    if columns_with_nonfinite:
        report["status"] = "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the configured Group 1 QDA dataset contract without training."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qda_static_k50.yaml"),
        help="Path to the QDA YAML configuration.",
    )
    args = parser.parse_args(argv)
    try:
        config, _, root = load_config(args.config)
        report = validate_parquet_dataset(config, root)
    except (DataContractError, FileNotFoundError) as exc:
        parser.exit(2, f"QDA data check failed: {exc}\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())