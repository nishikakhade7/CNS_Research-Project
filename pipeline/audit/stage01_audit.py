"""Stage 01 — Data Audit.

Inspects the actual files currently on disk (never trusts hardcoded counts from prior
documentation) and produces a JSON audit report: inventory, schema, row/feature counts,
label counts, dtype summary, missingness, duplicate audit, constant/zero-variance audit,
and provenance. All large files are processed in chunks — nothing here loads a full
multi-hundred-MB / multi-GB file into memory at once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.core.paths import resolve
from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage01_audit")

CHUNK_ROWS = 2000


def _file_sha256_prefix(path: Path, n_bytes: int = 8_000_000) -> str:
    """Hash the first n_bytes of a file — a cheap provenance fingerprint for huge files
    (a full-file hash of a 1.8GB CSV is not worth the I/O cost here; this still detects
    "this is not the same file" for practical purposes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def _audit_maldroid_dynamic(path: Path, label_field: str = "Class") -> dict[str, Any]:
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    feat_cols = [c for c in df.columns if c != label_field]
    label_counts = df[label_field].value_counts().sort_index().to_dict()
    zero_var = int((df[feat_cols].std(numeric_only=True) == 0).sum())
    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    return {
        "rows": int(n_rows),
        "feature_columns": len(feat_cols),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "missing_values_total": n_missing,
        "zero_variance_feature_columns": zero_var,
        "exact_duplicate_rows": n_dupes,
        "file_fingerprint_sha256_prefix": _file_sha256_prefix(path),
        "file_size_bytes": path.stat().st_size,
    }


def _audit_maldroid_static_repaired(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(str(path))
    n_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names
    n_cols = len(schema_names)
    label_col = "Class" if "Class" in schema_names else None

    label_counts: dict[str, int] = {}
    if label_col:
        col_table = pf.read(columns=[label_col])
        vc = pd.Series(col_table.column(label_col).to_pylist()).value_counts().sort_index()
        label_counts = {str(k): int(v) for k, v in vc.items()}

    # chunked column-family scan for text-vs-numeric split (streamed in column batches,
    # vectorized across each batch instead of looping pd.to_numeric per column — the
    # per-column Python loop was measured to be too slow across ~50,000 columns)
    batch_size = 4000
    feat_cols = [c for c in schema_names if c not in ("Class",)]
    n_string_like = 0
    n_numeric_like = 0
    sample_rows = min(2000, n_rows)
    n_batches = (len(feat_cols) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(feat_cols), batch_size)):
        batch_cols = feat_cols[start:start + batch_size]
        tbl = pf.read(columns=batch_cols)
        pdf = tbl.to_pandas().head(sample_rows)
        # the repaired parquet stores absent/blank flags as literal '' (empty string), not
        # NaN (Doc 02 A.9 repair used keep_default_na=False to preserve exact raw values) —
        # without this, every sparse binary column looks like it "fails" numeric parsing on
        # its blank cells and gets misclassified as text. Treat '' as missing here, same as
        # Stage 03's typing must (and does, after this fix).
        pdf = pdf.replace('', np.nan)

        notna_counts = pdf.notna().sum(axis=0)
        coerced = pdf.apply(pd.to_numeric, errors="coerce")
        coerced_notna_counts = coerced.notna().sum(axis=0)

        all_blank = notna_counts == 0
        numeric_like = (~all_blank) & (coerced_notna_counts >= 0.95 * notna_counts)
        n_numeric_like += int(all_blank.sum()) + int(numeric_like.sum())
        n_string_like += int((~all_blank & ~numeric_like).sum())
        log.info(f"  static typing scan: batch {bi+1}/{n_batches} done "
                 f"({len(batch_cols)} cols, running totals numeric={n_numeric_like} text={n_string_like})")

    return {
        "rows": int(n_rows),
        "feature_columns": n_cols - (1 if label_col else 0),
        "label_counts": label_counts,
        "columns_numeric_like_sampled": n_numeric_like,
        "columns_text_like_sampled": n_string_like,
        "provenance": "repaired from feature_vectors_static.csv (UTF-7 decode + header realignment "
                      "+ Class attached via ROW-ORDER alignment with the dynamic file — "
                      "NOT a hash-based join; MalDroid static files carry no sample hash)",
        "file_fingerprint_sha256_prefix": _file_sha256_prefix(path),
        "file_size_bytes": path.stat().st_size,
    }


def _audit_andmal_static_dir(dir_path: Path) -> dict[str, Any]:
    per_file: dict[str, Any] = {}
    all_hashes: dict[str, list[str]] = {}
    for csv_path in sorted(dir_path.glob("*.csv")):
        label = csv_path.stem
        n_rows = 0
        n_cols = None
        hashes_this_file: set[str] = set()
        n_string_like = None
        for chunk in pd.read_csv(csv_path, header=None, chunksize=CHUNK_ROWS, dtype=str, low_memory=False):
            n_rows += len(chunk)
            if n_cols is None:
                n_cols = chunk.shape[1]
            if n_string_like is None:
                sample = chunk.iloc[:, 1:].head(200)
                coerced = sample.apply(pd.to_numeric, errors="coerce")
                n_string_like = int((coerced.isna() & sample.notna()).any(axis=0).sum())
            hashes_this_file.update(chunk.iloc[:, 0].tolist())
        for h in hashes_this_file:
            all_hashes.setdefault(h, []).append(label)
        per_file[label] = {
            "rows": n_rows,
            "columns": n_cols,
            "feature_columns": (n_cols - 1) if n_cols else None,
            "unique_hashes": len(hashes_this_file),
            "internal_duplicate_hashes": n_rows - len(hashes_this_file),
            "columns_with_non_numeric_values_sampled": n_string_like,
        }
        log.info(f"  audited AndMal file {csv_path.name}: {n_rows} rows")

    overlap_pairs: dict[str, int] = {}
    for h, labels in all_hashes.items():
        if len(set(labels)) > 1:
            key = " + ".join(sorted(set(labels)))
            overlap_pairs[key] = overlap_pairs.get(key, 0) + 1

    return {
        "per_file": per_file,
        "total_rows": sum(v["rows"] for v in per_file.values()),
        "total_unique_hashes": len(all_hashes),
        "cross_file_hash_overlaps": overlap_pairs,
        "note": "headerless files — column 0 is the sample hash; there is NO label column, "
                "label is derived from the source filename only",
    }


def _audit_andmal_dynamic_dir(dir_path: Path) -> dict[str, Any]:
    per_file: dict[str, Any] = {}
    for csv_path in sorted(dir_path.glob("*.csv")):
        df_head = pd.read_csv(csv_path, nrows=5)
        n_cols = df_head.shape[1]
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            n_rows = sum(1 for _ in f) - 1
        per_file[csv_path.stem] = {
            "rows": n_rows,
            "columns": n_cols,
            "has_hash_category_family": all(c in df_head.columns for c in ("Hash", "Category", "Family")),
        }
    return {
        "per_file": per_file,
        "total_rows": sum(v["rows"] for v in per_file.values()),
        "note": "malicious categories only — no benign dynamic file exists in this release; "
                "before/after files are NOT row-aligned, only Hash-joinable",
    }


class DataAuditStage(Stage):
    name = "stage01_audit"
    depends_on: list[str] = []

    def config_section(self, cfg: dict) -> dict:
        return {"paths": cfg["paths"]}

    def run(self, ctx: StageContext) -> dict:
        cfg = ctx.cfg
        report: dict[str, Any] = {"maldroid": {}, "andmal": {}}

        log.info("Auditing MalDroid2020 dynamic (139-feature) file...")
        report["maldroid"]["dynamic_139"] = _audit_maldroid_dynamic(resolve(cfg["paths"]["maldroid_dynamic_139"]))

        log.info("Auditing MalDroid2020 dynamic (470-feature) file...")
        report["maldroid"]["dynamic_470"] = _audit_maldroid_dynamic(resolve(cfg["paths"]["maldroid_dynamic_470"]))

        static_repaired = resolve(cfg["paths"]["maldroid_static_repaired"])
        if static_repaired.exists():
            log.info("Auditing MalDroid2020 repaired static parquet...")
            report["maldroid"]["static_repaired"] = _audit_maldroid_static_repaired(static_repaired)
        else:
            raise FileNotFoundError(
                f"Repaired static parquet not found at {static_repaired}. "
                "This pipeline does not silently re-derive it — it must exist from the "
                "documented repair procedure (Doc 02) before Stage 01 can audit it."
            )

        log.info("Auditing AndMal2020 static Benign files (streamed, chunked)...")
        report["andmal"]["static_benign"] = _audit_andmal_static_dir(resolve(cfg["paths"]["andmal_static_benign"]))

        log.info("Auditing AndMal2020 static Malicious files (streamed, chunked — this includes a 1.8GB file, may take a few minutes)...")
        report["andmal"]["static_malicious"] = _audit_andmal_static_dir(resolve(cfg["paths"]["andmal_static_malicious"]))

        log.info("Auditing AndMal2020 dynamic before/after-reboot files...")
        report["andmal"]["dynamic"] = _audit_andmal_dynamic_dir(resolve(cfg["paths"]["andmal_dynamic"]))

        out_path = ctx.run_dir / "reports" / "stage01_data_audit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        log.info(f"Audit report written: {out_path}")
        return {"report_path": str(out_path)}
