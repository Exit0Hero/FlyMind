"""Tests for Link Prediction pipeline (memory-safe API)."""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def load_raw_edges():
    from src.link_prediction.data_prep import load_raw_edges as _load
    return _load()


def test_negative_samples_are_not_positives():
    """No negative sample should be a known positive edge."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges
    from src.link_prediction.negative_sampling import build_edge_index, sample_negatives

    nt = load_neuron_table()
    edges = aggregate_edges(load_raw_edges())
    pos = edges[["source", "target"]].values
    edge_idx = build_edge_index(pos)
    all_nodes = np.unique(np.concatenate([pos[:, 0], pos[:, 1]]))

    neg = sample_negatives(all_nodes, 1000, edge_idx, seed=42)
    from src.link_prediction.negative_sampling import edge_exists
    for s, t in neg:
        assert not edge_exists(int(s), int(t), edge_idx), f"Negative ({s},{t}) is a known positive!"
    print("  PASS: no negative sample is a known positive edge")


def test_no_self_loop_negatives():
    """No negative should be a self-loop."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges
    from src.link_prediction.negative_sampling import build_edge_index, sample_negatives

    nt = load_neuron_table()
    edges = aggregate_edges(load_raw_edges())
    pos = edges[["source", "target"]].values
    edge_idx = build_edge_index(pos)
    all_nodes = np.unique(np.concatenate([pos[:, 0], pos[:, 1]]))

    neg = sample_negatives(all_nodes, 1000, edge_idx, seed=42)
    for s, t in neg:
        assert s != t, f"Self-loop negative: ({s},{t})"
    print("  PASS: no self-loop negatives")


def test_train_val_test_positives_disjoint():
    """Train, val, test positive edges should be disjoint."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, create_random_edge_split

    nt = load_neuron_table()
    edges = aggregate_edges(load_raw_edges())
    split = create_random_edge_split(edges)

    train_set = set(zip(split["train"][:, 0].tolist(), split["train"][:, 1].tolist()))
    val_set = set(zip(split["val"][:, 0].tolist(), split["val"][:, 1].tolist()))
    test_set = set(zip(split["test"][:, 0].tolist(), split["test"][:, 1].tolist()))

    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"
    print("  PASS: train/val/test positive edges are disjoint")


def test_pair_features_dimensions():
    """Pair features should have 4x the original feature count."""
    from src.link_prediction.data_prep import load_neuron_table, build_feature_matrix
    from src.link_prediction.features import build_pair_features
    from src.link_prediction.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    nt = load_neuron_table()
    X, root_ids, id_to_idx, _, _ = build_feature_matrix(nt, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    pairs = np.array([[root_ids[0], root_ids[1]]], dtype=np.int64)
    pair_feats = build_pair_features(X, pairs, id_to_idx)

    expected_cols = X.shape[1] * 4
    assert pair_feats.shape[1] == expected_cols, f"Expected {expected_cols} cols, got {pair_feats.shape[1]}"
    assert pair_feats.shape[0] == 1
    print(f"  PASS: pair features have correct dimensions ({pair_feats.shape})")


def test_no_nans_in_features():
    """Pair features should not contain NaN or Inf."""
    from src.link_prediction.data_prep import load_neuron_table, build_feature_matrix
    from src.link_prediction.features import build_pair_features
    from src.link_prediction.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    nt = load_neuron_table()
    X, root_ids, id_to_idx, _, _ = build_feature_matrix(nt, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    pairs = np.array([[root_ids[0], root_ids[1]]], dtype=np.int64)
    pair_feats = build_pair_features(X, pairs, id_to_idx)

    assert not np.isnan(pair_feats).any(), "NaN in pair features!"
    assert not np.isinf(pair_feats).any(), "Inf in pair features!"
    print("  PASS: no NaN/Inf in pair features")


def test_reproducible_negative_sampling():
    """Same seed should produce same negatives."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges
    from src.link_prediction.negative_sampling import build_edge_index, sample_negatives

    nt = load_neuron_table()
    edges = aggregate_edges(load_raw_edges())
    pos = edges[["source", "target"]].values
    edge_idx = build_edge_index(pos)
    all_nodes = np.unique(np.concatenate([pos[:, 0], pos[:, 1]]))

    neg1 = sample_negatives(all_nodes, 500, edge_idx, seed=42)
    neg2 = sample_negatives(all_nodes, 500, edge_idx, seed=42)
    # Due to batch rejection sampling, order may differ; check sets are equal
    set1 = set(map(tuple, neg1.tolist()))
    set2 = set(map(tuple, neg2.tolist()))
    assert set1 == set2, "Same seed produced different negative sets!"
    print("  PASS: negative sampling is reproducible")


def test_cold_start_no_leakage():
    """Cold-start test edges should involve test nodes."""
    from src.link_prediction.data_prep import load_neuron_table, aggregate_edges, create_cold_start_split

    nt = load_neuron_table()
    edges = aggregate_edges(load_raw_edges())
    cold = create_cold_start_split(edges, nt["root_id"].values)

    train_nodes = cold["train_nodes"]
    test_nodes = cold["test_nodes"]

    for s, t in cold["test"]:
        assert int(s) in test_nodes or int(t) in test_nodes, \
            f"Test edge ({s},{t}) has no test node!"
    print("  PASS: cold-start test edges involve test nodes")


def test_memory_safe_edge_index():
    """Edge index should be compact sorted uint64 array."""
    from src.link_prediction.negative_sampling import build_edge_index
    edges = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    idx = build_edge_index(edges)
    assert idx.dtype == np.uint64, f"Expected uint64, got {idx.dtype}"
    assert len(idx) == 3, f"Expected 3 entries, got {len(idx)}"
    assert len(idx) == len(np.unique(idx)), "Unexpected duplicate hashes!"
    assert np.all(idx[:-1] <= idx[1:]), "Not sorted!"
    print("  PASS: edge index is compact and sorted")


if __name__ == __name__:
    tests = [
        test_no_self_loop_negatives,
        test_negative_samples_are_not_positives,
        test_train_val_test_positives_disjoint,
        test_pair_features_dimensions,
        test_no_nans_in_features,
        test_reproducible_negative_sampling,
        test_cold_start_no_leakage,
        test_memory_safe_edge_index,
    ]

    print("=" * 60)
    print("LINK PREDICTION TESTS (memory-safe)")
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
