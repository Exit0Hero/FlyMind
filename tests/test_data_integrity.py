#!/usr/bin/env python3
"""Tests for FlyMind processed data integrity."""

import pathlib
import sys

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"


def test_neuron_table_loads():
    """Neuron table parquet can be loaded."""
    path = DATA_DIR / "neuron_table.parquet"
    assert path.exists(), f"Missing: {path}"
    df = pq.read_table(path).to_pandas()
    assert len(df) > 100_000, f"Too few rows: {len(df)}"
    print(f"  PASS: neuron_table loads ({len(df)} rows, {len(df.columns)} cols)")


def test_neuron_ids_unique():
    """root_id is unique in the neuron table."""
    df = pq.read_table(DATA_DIR / "neuron_table.parquet").to_pandas()
    dup_count = df["root_id"].duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate root_ids"
    print(f"  PASS: root_id unique ({len(df)} unique neurons)")


def test_neuron_ids_are_integers():
    """root_id values are valid integers."""
    df = pq.read_table(DATA_DIR / "neuron_table.parquet").to_pandas()
    assert pd.api.types.is_integer_dtype(df["root_id"]), f"root_id dtype is {df['root_id'].dtype}"
    assert df["root_id"].notna().all(), "root_id has nulls"
    print(f"  PASS: root_id is integer type ({df['root_id'].dtype})")


def test_edge_table_loads():
    """Edge table parquet can be loaded."""
    for name in ["edges_princeton.parquet", "edges_buhmann.parquet"]:
        path = DATA_DIR / name
        assert path.exists(), f"Missing: {path}"
        df = pq.read_table(path).to_pandas()
        assert len(df) > 0, f"Empty: {name}"
        assert set(df.columns) == {"source", "target", "weight"}, f"Unexpected columns: {df.columns.tolist()}"
        print(f"  PASS: {name} loads ({len(df)} edges)")


def test_edge_no_self_loops():
    """Edge tables have no self-loops."""
    for name in ["edges_princeton.parquet", "edges_buhmann.parquet"]:
        df = pq.read_table(DATA_DIR / name).to_pandas()
        self_loops = (df["source"] == df["target"]).sum()
        assert self_loops == 0, f"{name}: {self_loops} self-loops"
        print(f"  PASS: {name} no self-loops")


def test_edge_weights_positive():
    """All edge weights are positive integers."""
    for name in ["edges_princeton.parquet", "edges_buhmann.parquet"]:
        df = pq.read_table(DATA_DIR / name).to_pandas()
        assert (df["weight"] > 0).all(), f"{name}: has zero/negative weights"
        assert pd.api.types.is_integer_dtype(df["weight"]), f"{name}: weight not integer"
        print(f"  PASS: {name} weights positive integers")


def test_edge_source_target_are_integers():
    """Source and target are integer types."""
    for name in ["edges_princeton.parquet", "edges_buhmann.parquet"]:
        df = pq.read_table(DATA_DIR / name).to_pandas()
        assert pd.api.types.is_integer_dtype(df["source"]), f"{name}: source not int"
        assert pd.api.types.is_integer_dtype(df["target"]), f"{name}: target not int"
        print(f"  PASS: {name} source/target are integers")


def test_edge_coverage():
    """Most edge source/target IDs exist in neuron table."""
    neuron_df = pq.read_table(DATA_DIR / "neuron_table.parquet").to_pandas()
    neuron_ids = set(neuron_df["root_id"])

    for name in ["edges_princeton.parquet"]:
        edge_df = pq.read_table(DATA_DIR / name).to_pandas()
        sources_in = edge_df["source"].isin(neuron_ids).sum()
        targets_in = edge_df["target"].isin(neuron_ids).sum()
        src_pct = 100 * sources_in / len(edge_df)
        tgt_pct = 100 * targets_in / len(edge_df)
        print(f"  INFO: {name} source coverage: {src_pct:.1f}%, target coverage: {tgt_pct:.1f}%")
        assert src_pct > 90, f"Source coverage too low: {src_pct:.1f}%"
        assert tgt_pct > 90, f"Target coverage too low: {tgt_pct:.1f}%"
        print(f"  PASS: {name} coverage > 90%")


def test_no_data_type_corruption():
    """Key columns have expected types in neuron table."""
    df = pq.read_table(DATA_DIR / "neuron_table.parquet").to_pandas()
    assert pd.api.types.is_integer_dtype(df["root_id"]), "root_id corrupted"
    assert pd.api.types.is_numeric_dtype(df["nt_type_score"]), "nt_type_score corrupted"
    assert pd.api.types.is_numeric_dtype(df["length_nm"]), "length_nm corrupted"
    print("  PASS: No data type corruption detected")


def main():
    tests = [
        test_neuron_table_loads,
        test_neuron_ids_unique,
        test_neuron_ids_are_integers,
        test_edge_table_loads,
        test_edge_no_self_loops,
        test_edge_weights_positive,
        test_edge_source_target_are_integers,
        test_edge_coverage,
        test_no_data_type_corruption,
    ]

    print("=" * 60)
    print("FLYWIRE DATA INTEGRITY TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
