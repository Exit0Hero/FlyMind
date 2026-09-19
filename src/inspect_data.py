#!/usr/bin/env python3
"""Comprehensive audit of FlyWire FAFB dataset files."""

import os
import sys
import gzip
import time
import json
import pathlib
from collections import defaultdict

import pandas as pd
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset"
REPORT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_file_info(path: pathlib.Path) -> dict:
    """Basic file metadata."""
    stat = path.stat()
    ext = path.suffix
    if path.name.endswith(".csv.gz"):
        fmt = "csv.gz"
    elif path.name.endswith(".tsv.gz"):
        fmt = "tsv.gz"
    elif path.name.endswith(".parquet"):
        fmt = "parquet"
    else:
        fmt = ext.lstrip(".")
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / 1e6, 2),
        "format": fmt,
    }


def read_csv_gz_sample(path: pathlib.Path, nrows: int = 5) -> pd.DataFrame:
    """Read first nrows from gzipped CSV."""
    return pd.read_csv(
        path, nrows=nrows, compression="gzip",
        dtype_backend="numpy_nullable",
    )


def count_rows_chunked(path: pathlib.Path, chunksize: int = 500_000) -> int:
    """Count rows without loading entire file."""
    count = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, compression="gzip", dtype_backend="numpy_nullable"):
        count += len(chunk)
    return count


def audit_single_file(path: pathlib.Path) -> dict:
    """Full audit of a single CSV.gz file."""
    info = get_file_info(path)
    print(f"  Auditing {info['filename']} ({info['size_mb']} MB)...")

    t0 = time.time()

    # Sample for schema / value analysis
    try:
        sample = read_csv_gz_sample(path, nrows=5000)
    except Exception as e:
        return {**info, "error": str(e)}

    info["n_columns"] = len(sample.columns)
    info["column_names"] = list(sample.columns)
    info["dtypes"] = {c: str(sample[c].dtype) for c in sample.columns}
    # Convert sample rows to JSON-safe dicts
    sample_rows = sample.head(5).to_dict(orient="records")
    info["sample_rows"] = [
        {k: (str(v) if isinstance(v, (np.integer, np.floating, pd.Timestamp)) else v)
         for k, v in row.items()}
        for row in sample_rows
    ]

    # Missing values (from sample)
    missing_counts = sample.isnull().sum()
    info["missing_counts"] = {c: int(v) for c, v in missing_counts.items()}
    info["missing_pct"] = {
        c: round(100 * v / len(sample), 2)
        for c, v in missing_counts.items()
    }

    # Unique values from sample
    info["unique_in_sample"] = {
        c: int(sample[c].nunique()) for c in sample.columns
    }

    # Value frequencies for low-cardinality columns
    value_freqs = {}
    for c in sample.columns:
        nuniq = sample[c].nunique()
        if nuniq <= 20:
            vc = sample[c].value_counts().head(10)
            value_freqs[c] = {str(k): int(v) for k, v in vc.items()}
    info["value_frequencies"] = value_freqs

    # Full row count
    info["total_rows"] = count_rows_chunked(path)

    # Full unique counts for likely ID columns
    id_candidates = [
        c for c in sample.columns
        if "id" in c.lower() or "root_id" in c.lower()
    ]
    if id_candidates:
        unique_counts = {}
        for c in id_candidates:
            uniq = set()
            for chunk in pd.read_csv(path, usecols=[c], chunksize=500_000, compression="gzip", dtype_backend="numpy_nullable"):
                uniq.update(chunk[c].dropna().values)
            unique_counts[c] = len(uniq)
        info["unique_id_columns"] = unique_counts

    # Duplicate rows (sample only for speed)
    info["duplicate_rows_in_sample"] = int(sample.duplicated().sum())

    # Likely column categories
    likely_ids = []
    likely_cats = []
    likely_nums = []
    _string_types = {"object", "string", "string[python]", "string[pyarrow]"}
    _int_types = {"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"}
    _float_types = {"float16", "float32", "float64", "Float16", "Float32", "Float64"}
    for c in sample.columns:
        dt_str = str(sample[c].dtype)
        if "id" in c.lower():
            likely_ids.append(c)
        elif dt_str in _string_types and sample[c].nunique() < 50:
            likely_cats.append(c)
        elif dt_str in _int_types or dt_str in _float_types:
            likely_nums.append(c)

    info["likely_id_columns"] = likely_ids
    info["likely_categorical_columns"] = likely_cats
    info["likely_numerical_columns"] = likely_nums

    elapsed = round(time.time() - t0, 2)
    info["audit_time_seconds"] = elapsed

    return info


def infer_neuron_identifier(audits: dict) -> dict:
    """Determine which column is the neuron identifier across files."""
    id_col_presence = defaultdict(list)
    for fname, audit in audits.items():
        if "error" in audit:
            continue
        for col in audit.get("column_names", []):
            if col == "root_id":
                id_col_presence["root_id"].append(fname)
            elif col == "body_id":
                id_col_presence["body_id"].append(fname)
    return dict(id_col_presence)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def main():
    print("=" * 70)
    print("FLYWIRE DATASET AUDIT")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")

    csv_files = sorted(DATA_DIR.glob("*.csv.gz"))
    other_files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.suffix in (".zip", ".md5", ".jpeg", ".png", ".pdf")
    )
    print(f"\nFound {len(csv_files)} CSV.gz files, {len(other_files)} other files")

    audits = {}
    for path in csv_files:
        try:
            audits[path.name] = audit_single_file(path)
        except Exception as e:
            audits[path.name] = {"filename": path.name, "error": str(e)}
            print(f"  ERROR auditing {path.name}: {e}")

    # Identifier analysis
    print("\n--- Identifier Analysis ---")
    id_presence = infer_neuron_identifier(audits)
    for col, files in id_presence.items():
        print(f"  {col} found in: {files}")

    # Save full audit as JSON
    report_path = REPORT_DIR / "file_audit.json"
    with open(report_path, "w") as f:
        json.dump(audits, f, indent=2, default=str, cls=NumpyEncoder)
    print(f"\nFull audit saved to: {report_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'File':<45} {'Rows':>12} {'Cols':>5} {'Size_MB':>8}")
    print("-" * 70)
    for fname, audit in audits.items():
        rows = audit.get("total_rows", "?")
        cols = audit.get("n_columns", "?")
        size = audit.get("size_mb", "?")
        print(f"{fname:<45} {rows:>12} {cols:>5} {size:>8}")

    return audits


if __name__ == "__main__":
    audits = main()
