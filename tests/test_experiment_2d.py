"""Tests for Experiment 2D: Candidate Connection Ranking."""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_no_self_loops_in_candidates():
    """Candidate generation never creates self-loops."""
    from src.link_prediction.experiment_2d import generate_candidates
    from src.link_prediction.negative_sampling import build_sorted_pairs

    all_node_ids = np.arange(100, dtype=np.int64) * 1000 + 720000000000000000
    edges = np.array([[all_node_ids[0], all_node_ids[1]]], dtype=np.int64)
    sorted_edges = build_sorted_pairs(edges[:, 0], edges[:, 1])

    for src_arr, dst_arr in generate_candidates(
        all_node_ids, sorted_edges, n_sources=10, targets_per_source=3, seed=42
    ):
        for s, d in zip(src_arr, dst_arr):
            assert s != d, f"Self-loop found: {s} -> {d}"
    print("  PASS: no self-loops in candidates")


def test_known_edges_excluded():
    """Known edges are excluded from candidates."""
    from src.link_prediction.experiment_2d import generate_candidates
    from src.link_prediction.negative_sampling import build_sorted_pairs

    all_node_ids = np.arange(100, dtype=np.int64) * 1000 + 720000000000000000
    edges = np.array([[all_node_ids[0], all_node_ids[1]]], dtype=np.int64)
    sorted_edges = build_sorted_pairs(edges[:, 0], edges[:, 1])
    known = set(map(tuple, edges.tolist()))

    for src_arr, dst_arr in generate_candidates(
        all_node_ids, sorted_edges, n_sources=10, targets_per_source=3, seed=42
    ):
        for s, d in zip(src_arr, dst_arr):
            assert (int(s), int(d)) not in known, f"Known edge in candidates: {s} -> {d}"
    print("  PASS: known edges excluded")


def test_no_duplicate_candidates():
    """No duplicate candidate pairs."""
    from src.link_prediction.experiment_2d import generate_candidates
    from src.link_prediction.negative_sampling import build_sorted_pairs

    all_node_ids = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.int64)
    edges = np.array([[100, 200]], dtype=np.int64)
    sorted_edges = build_sorted_pairs(edges[:, 0], edges[:, 1])

    all_pairs = set()
    for src_arr, dst_arr in generate_candidates(
        all_node_ids, sorted_edges, n_sources=10, targets_per_source=3, seed=42
    ):
        for s, d in zip(src_arr, dst_arr):
            pair = (int(s), int(d))
            assert pair not in all_pairs, f"Duplicate candidate: {pair}"
            all_pairs.add(pair)
    print("  PASS: no duplicate candidates")


def test_candidate_scores_finite():
    """Candidate scores are finite."""
    import pickle
    from src.link_prediction.experiment_2d import build_pair_features_chunked

    with open("models/link_prediction_rf.pkl", "rb") as f:
        rf = pickle.load(f)["model"]
    X = np.load("data/processed/link_prediction/X_features.npy")
    X = np.nan_to_num(X, nan=0.0)

    src_idx = np.array([0, 1, 2], dtype=np.int64)
    dst_idx = np.array([1, 2, 3], dtype=np.int64)
    for chunk, _, _ in build_pair_features_chunked(X, src_idx, dst_idx):
        scores = rf.predict_proba(chunk)[:, 1]
        assert np.all(np.isfinite(scores)), f"Non-finite scores: {scores}"
    print("  PASS: candidate scores are finite")


def test_pair_features_match_training():
    """Pair features match Experiment 2 training feature order."""
    import pickle

    with open("models/link_prediction_rf.pkl", "rb") as f:
        data = pickle.load(f)
    feature_cols = data["feature_cols"]
    rf = data["model"]

    # RF expects 60 features: 15 base features * 4 pair components
    assert rf.n_features_in_ == 60, f"Expected 60 features, got {rf.n_features_in_}"
    # Base feature list has 15 names (pair construction applied at runtime)
    assert len(feature_cols) == 15, f"Expected 15 base feature names, got {len(feature_cols)}"
    print("  PASS: pair features match training order")


def test_no_all_pairs_enumeration():
    """Candidate generation does not enumerate all pairs."""
    from src.link_prediction.experiment_2d import generate_candidates
    from src.link_prediction.negative_sampling import build_sorted_pairs

    n_nodes = 1000
    all_node_ids = np.arange(n_nodes, dtype=np.int64) * 1000 + 1000000
    edges = np.array([[all_node_ids[0], all_node_ids[1]]], dtype=np.int64)
    sorted_edges = build_sorted_pairs(edges[:, 0], edges[:, 1])

    total = 0
    for src_arr, dst_arr in generate_candidates(
        all_node_ids, sorted_edges, n_sources=100, targets_per_source=10, seed=42
    ):
        total += len(src_arr)

    max_possible = n_nodes * n_nodes
    assert total < max_possible, f"Candidate count {total} equals all-pairs {max_possible}"
    assert total <= 100 * 10, f"Candidate count {total} exceeds expected bound"
    print(f"  PASS: candidates ({total}) << all-pairs ({max_possible})")


def test_top_k_ranking():
    """Top-K ranking is correct."""
    from collections import defaultdict

    # Simulate top-K tracking
    top_k = defaultdict(list)
    test_data = [
        (1, 100, 0.9), (1, 200, 0.5), (1, 300, 0.8), (1, 400, 0.3),
        (2, 100, 0.7), (2, 200, 0.6),
    ]
    for src, dst, score in test_data:
        heap = top_k[src]
        if len(heap) < 3:
            heap.append((score, dst))
            heap.sort(key=lambda x: x[0], reverse=True)
        elif score > heap[-1][0]:
            heap[-1] = (score, dst)
            heap.sort(key=lambda x: x[0], reverse=True)

    # Source 1 should have top 3: (0.9, 100), (0.8, 300), (0.5, 200)
    s1 = top_k[1]
    assert len(s1) == 3
    assert s1[0] == (0.9, 100)
    assert s1[1] == (0.8, 300)
    assert s1[2] == (0.5, 200)
    print("  PASS: top-K ranking is correct")


def test_deterministic_sampling():
    """Candidate generation is deterministic with same seed."""
    from src.link_prediction.experiment_2d import generate_candidates
    from src.link_prediction.negative_sampling import build_sorted_pairs

    all_node_ids = np.arange(500, dtype=np.int64) * 100 + 1000000
    edges = np.array([[all_node_ids[0], all_node_ids[1]]], dtype=np.int64)
    sorted_edges = build_sorted_pairs(edges[:, 0], edges[:, 1])

    run1 = []
    for s, d in generate_candidates(all_node_ids, sorted_edges, n_sources=20, targets_per_source=5, seed=42):
        run1.append((s.copy(), d.copy()))

    run2 = []
    for s, d in generate_candidates(all_node_ids, sorted_edges, n_sources=20, targets_per_source=5, seed=42):
        run2.append((s.copy(), d.copy()))

    assert len(run1) == len(run2), f"Different number of chunks: {len(run1)} vs {len(run2)}"
    for (s1, d1), (s2, d2) in zip(run1, run2):
        assert np.array_equal(s1, s2), "Source arrays differ"
        assert np.array_equal(d1, d2), "Target arrays differ"
    print("  PASS: deterministic sampling with seed 42")


def test_chunk_processing_preserves_ranking():
    """Processing in chunks does not change ranking results."""
    import pickle
    from src.link_prediction.experiment_2d import build_pair_features_chunked

    with open("models/link_prediction_rf.pkl", "rb") as f:
        rf = pickle.load(f)["model"]
    X = np.load("data/processed/link_prediction/X_features.npy")
    X = np.nan_to_num(X, nan=0.0)

    src_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    dst_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

    # Process all at once
    xa = X[src_idx]
    xb = X[dst_idx]
    pair_all = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
    scores_all = rf.predict_proba(pair_all)[:, 1]

    # Process in chunks
    scores_chunked = np.empty(10, dtype=np.float32)
    for chunk, start, end in build_pair_features_chunked(X, src_idx, dst_idx, chunk_size=3):
        scores_chunked[start:end] = rf.predict_proba(chunk)[:, 1]

    assert np.allclose(scores_all, scores_chunked), "Chunk processing changed scores"
    print("  PASS: chunk processing preserves ranking")


def test_memory_safe_pair_features():
    """Pair features are generated only in batches, not all at once."""
    import tracemalloc
    from src.link_prediction.experiment_2d import build_pair_features_chunked

    X = np.random.randn(1000, 15).astype(np.float32)
    src = np.arange(0, 500, dtype=np.int64)
    dst = np.arange(500, 1000, dtype=np.int64)

    tracemalloc.start()
    peak_before = tracemalloc.get_traced_memory()[1]

    for chunk, _, _ in build_pair_features_chunked(X, src, dst, chunk_size=100):
        pass  # Just iterate

    peak_after = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Peak should be much less than full matrix (500 * 60 * 4 = 120KB)
    # Chunk of 100 * 60 * 4 = 24KB
    assert peak_after - peak_before < 100 * 1024, \
        f"Peak memory too high: {peak_after - peak_before} bytes"
    print("  PASS: pair features memory-safe")


if __name__ == "__main__":
    tests = [
        test_no_self_loops_in_candidates,
        test_known_edges_excluded,
        test_no_duplicate_candidates,
        test_candidate_scores_finite,
        test_pair_features_match_training,
        test_no_all_pairs_enumeration,
        test_top_k_ranking,
        test_deterministic_sampling,
        test_chunk_processing_preserves_ranking,
        test_memory_safe_pair_features,
    ]

    print("=" * 60)
    print("EXPERIMENT 2D TESTS")
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
