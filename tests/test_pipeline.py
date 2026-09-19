"""Tests for the ML pipeline."""

import numpy as np
import pandas as pd
from src.config import SEED, TARGET_COL, ALL_FEATURES


def test_data_loads():
    from src.data_loader import load_neuron_table, load_and_aggregate_edges
    df = load_neuron_table()
    edges = load_and_aggregate_edges()
    assert len(df) == 139255
    assert len(edges) > 0
    assert set(edges.columns) == {"source", "target", "weight"}
    print(f"  PASS: data loads ({len(df)} nodes, {len(edges)} edges)")


def test_edge_aggregation():
    from src.data_loader import load_and_aggregate_edges
    edges = load_and_aggregate_edges()
    # No duplicate (source, target) pairs
    dupes = edges.duplicated(subset=["source", "target"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate pairs after aggregation"
    # No self-loops
    self_loops = (edges["source"] == edges["target"]).sum()
    assert self_loops == 0, f"Found {self_loops} self-loops"
    # All weights positive
    assert (edges["weight"] > 0).all()
    print(f"  PASS: edge aggregation ({len(edges)} unique pairs, no dupes/self-loops)")


def test_target_leakage():
    """Verify excluded features are not used."""
    from src.config import EXCLUDED_FEATURES, SAFE_FEATURES_NUMERIC, SAFE_FEATURES_CATEGORICAL
    from src.data_loader import load_neuron_table
    df = load_neuron_table()

    # Verify target exists
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].notna().all()

    # Verify safe features exist
    for f in SAFE_FEATURES_NUMERIC:
        assert f in df.columns, f"Missing safe numeric feature: {f}"
    for f in SAFE_FEATURES_CATEGORICAL:
        assert f in df.columns, f"Missing safe categorical feature: {f}"

    # Verify excluded features are NOT in safe list
    overlap = set(EXCLUDED_FEATURES) & set(SAFE_FEATURES_NUMERIC + SAFE_FEATURES_CATEGORICAL)
    assert len(overlap) == 0, f"Excluded features leak into safe list: {overlap}"

    print(f"  PASS: no target leakage ({len(SAFE_FEATURES_NUMERIC)} numeric, {len(SAFE_FEATURES_CATEGORICAL)} categorical)")


def test_train_val_test_split():
    from src.data_loader import load_neuron_table
    from src.preprocessing import create_split
    df = load_neuron_table()
    idx_train, idx_val, idx_test = create_split(df)

    total = len(idx_train) + len(idx_val) + len(idx_test)
    assert total == len(df), f"Split size {total} != data size {len(df)}"

    # Check no overlap
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0

    # Check approximate ratios
    train_pct = len(idx_train) / total
    val_pct = len(idx_val) / total
    test_pct = len(idx_test) / total
    assert 0.68 < train_pct < 0.72, f"Train ratio: {train_pct}"
    assert 0.13 < val_pct < 0.17, f"Val ratio: {val_pct}"
    assert 0.13 < test_pct < 0.17, f"Test ratio: {test_pct}"

    print(f"  PASS: split ({len(idx_train)} train, {len(idx_val)} val, {len(idx_test)} test)")


def test_preprocessing():
    from src.data_loader import load_neuron_table
    from src.preprocessing import Preprocessor, create_split
    df = load_neuron_table()
    idx_train, idx_val, idx_test = create_split(df)

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    pp = Preprocessor()
    X, y = pp.fit_transform(df, train_mask)

    assert X.shape[0] == len(df)
    assert y.shape[0] == len(df)
    assert not np.isnan(X).any(), "NaN in features after preprocessing"
    assert X.dtype == np.float32

    print(f"  PASS: preprocessing ({X.shape[1]} features, no NaN)")


def test_graph_construction():
    from src.data_loader import load_neuron_table, load_and_aggregate_edges
    from src.preprocessing import Preprocessor, create_split
    from src.graph_builder import build_graph
    from src.config import SEED
    df = load_neuron_table()
    edges = load_and_aggregate_edges()
    idx_train, idx_val, idx_test = create_split(df)

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx_train] = True

    pp = Preprocessor()
    X, y = pp.fit_transform(df, train_mask)

    node_ids = df["root_id"].values
    edge_weights = edges["weight"].values
    edges_arr = edges[["source", "target"]].values

    train_mask_arr = np.zeros(len(df), dtype=bool)
    val_mask_arr = np.zeros(len(df), dtype=bool)
    test_mask_arr = np.zeros(len(df), dtype=bool)
    train_mask_arr[idx_train] = True
    val_mask_arr[idx_val] = True
    test_mask_arr[idx_test] = True

    data = build_graph(X, y, edges_arr, node_ids, edge_weights,
                       train_mask_arr, val_mask_arr, test_mask_arr)

    assert data.num_nodes == len(df)
    assert data.x.shape[1] == X.shape[1]
    assert data.edge_index.shape[0] == 2
    # With reverse edges, should have 2x original
    assert data.edge_index.shape[1] == len(edges) * 2

    print(f"  PASS: graph construction ({data.num_nodes} nodes, {data.edge_index.shape[1]} edges)")


if __name__ == "__main__":
    tests = [
        test_data_loads,
        test_edge_aggregation,
        test_target_leakage,
        test_train_val_test_split,
        test_preprocessing,
        test_graph_construction,
    ]

    print("=" * 60)
    print("ML PIPELINE TESTS")
    print("=" * 60)
    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n{test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
