#!/usr/bin/env python3
"""Experiment 2C: Memory-safe data loading."""

import gc
import os
import json
import numpy as np
from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges


def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def log_mem(tag=""):
    print(f"  [MEM:{tag}] RSS: {_rss_mb():.0f} MB")


def load_random_split(data_dir):
    X = np.load(os.path.join(data_dir, "X_features.npy"))
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    with open(os.path.join(data_dir, "id_to_idx.json")) as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    node_ids = np.array(sorted(id_to_idx.keys()), dtype=np.int64)

    splits = {}
    for key in ["train", "val", "test"]:
        pos = np.load(os.path.join(data_dir, f"split_{key}.npy"))
        neg = np.load(os.path.join(data_dir, f"{key}_neg.npy"))
        splits[key] = (pos, neg)
        log_mem(f"loaded {key}")

    n_feat = X.shape[1]
    print(f"  Nodes: {len(node_ids)}, Features: {n_feat}")
    return X, node_ids, id_to_idx, n_feat, splits


def load_cold_start_split(cold_dir, node_ids):
    train_edges = np.load(os.path.join(cold_dir, "split_train.npy"))
    val_edges = np.load(os.path.join(cold_dir, "split_val.npy"))
    test_edges = np.load(os.path.join(cold_dir, "split_test.npy"))
    test_neg = np.load(os.path.join(cold_dir, "test_neg.npy"))
    val_neg = np.load(os.path.join(cold_dir, "val_neg.npy"))
    log_mem("cold-start loaded")

    # Build unique node IDs from cold-start edges (not from random split)
    cold_node_ids = np.unique(np.concatenate([
        train_edges[:, 0], train_edges[:, 1],
        val_edges[:, 0], val_edges[:, 1],
        test_edges[:, 0], test_edges[:, 1],
    ]))

    all_s = np.concatenate([train_edges[:, 0], val_edges[:, 0], test_edges[:, 0]])
    all_t = np.concatenate([train_edges[:, 1], val_edges[:, 1], test_edges[:, 1]])
    sorted_pairs = build_sorted_pairs(all_s, all_t)
    del all_s, all_t
    gc.collect()

    n_train = len(train_edges)
    rng = np.random.default_rng(42)
    train_neg = _sample_negatives(sorted_pairs, cold_node_ids, n_train, rng)
    del sorted_pairs, cold_node_ids
    gc.collect()
    log_mem("cold-start ready")

    print(f"  Cold-start: Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
    return {
        "train": (train_edges, train_neg),
        "val": (val_edges, val_neg),
        "test": (test_edges, test_neg),
    }


def _sample_negatives(sorted_pairs, node_ids, n_neg, rng):
    """Sample negative edges not in sorted_pairs, using only valid node IDs."""
    n_nodes = len(node_ids)
    neg_s = np.empty(n_neg, dtype=np.int64)
    neg_t = np.empty(n_neg, dtype=np.int64)
    collected = 0
    while collected < n_neg:
        batch_size = int((n_neg - collected) * 1.5) + 1000
        cand_s = node_ids[rng.integers(0, n_nodes, size=batch_size)]
        cand_t = node_ids[rng.integers(0, n_nodes, size=batch_size)]
        valid = verify_not_in_edges(cand_s, cand_t, sorted_pairs)
        cs, ct = cand_s[valid], cand_t[valid]
        n_take = min(len(cs), n_neg - collected)
        if n_take > 0:
            neg_s[collected:collected + n_take] = cs[:n_take]
            neg_t[collected:collected + n_take] = ct[:n_take]
            collected += n_take
    return np.stack([neg_s, neg_t], axis=1)
