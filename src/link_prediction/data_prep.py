"""FlyMind Link Prediction — Data preparation, splits (memory-safe)."""

import gc
import json
import pathlib
import numpy as np
import pandas as pd

from src.link_prediction.config import (
    SEED, PROCESSED_DIR, LP_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    COLD_TRAIN_RATIO, COLD_VAL_RATIO, COLD_TEST_RATIO,
)


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------
def _rss_mb():
    """Return current RSS in MB (Linux only)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def log_mem(label):
    print(f"  [MEM] {label}: {_rss_mb():.0f} MB RSS")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_neuron_table():
    return pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")


def load_raw_edges():
    raw_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "dataset" / "connections_princeton.csv.gz"
    return pd.read_csv(raw_path, compression="gzip")


def aggregate_edges(raw_edges):
    """Aggregate neuropil-specific rows into one directed edge per pair."""
    agg = raw_edges.groupby(["pre_root_id", "post_root_id"]).agg(
        weight=("syn_count", "sum"),
        num_neuropils=("neuropil", "nunique"),
    ).reset_index()
    agg.columns = ["source", "target", "weight", "num_neuropils"]
    return agg


def build_feature_matrix(neuron_table, numeric_features, categorical_features):
    """Build feature matrix from neuron table."""
    df = neuron_table.copy()
    cat_encoders = {}
    for col in categorical_features:
        vals = df[col].fillna("__MISSING__").astype(str).unique()
        mapping = {v: i for i, v in enumerate(sorted(vals))}
        cat_encoders[col] = mapping
        df[col + "_enc"] = df[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype(np.float32)

    feature_cols = numeric_features + [c + "_enc" for c in categorical_features]
    X = df[feature_cols].values.astype(np.float32)
    # Fill any remaining NaN/Inf
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    root_ids = df["root_id"].values
    id_to_idx = {int(rid): i for i, rid in enumerate(root_ids)}
    return X, root_ids, id_to_idx, cat_encoders, feature_cols


# ---------------------------------------------------------------------------
# Edge splits
# ---------------------------------------------------------------------------
def create_random_edge_split(edges_df, seed=SEED):
    """Split positive edges into train/val/test (70/15/15)."""
    pos_edges = edges_df[["source", "target"]].values.copy()
    np.random.seed(seed)
    indices = np.arange(len(pos_edges))
    np.random.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": pos_edges[indices[:n_train]],
        "val": pos_edges[indices[n_train:n_train + n_val]],
        "test": pos_edges[indices[n_train + n_val:]],
    }


def create_cold_start_split(edges_df, neuron_ids, seed=SEED):
    """Split nodes into train/val/test, then assign edges.

    Uses vectorized numpy operations instead of Python loops.
    """
    np.random.seed(seed)
    unique_nodes = np.unique(neuron_ids)
    np.random.shuffle(unique_nodes)

    n = len(unique_nodes)
    n_train = int(n * COLD_TRAIN_RATIO)
    n_val = int(n * COLD_VAL_RATIO)

    train_set = set(unique_nodes[:n_train].tolist())
    val_set = set(unique_nodes[n_train:n_train + n_val].tolist())
    test_set = set(unique_nodes[n_train + n_val:].tolist())

    src = edges_df["source"].values
    tgt = edges_df["target"].values

    # Vectorized membership via numpy searchsorted on sorted arrays
    train_arr = np.array(sorted(train_set))
    val_arr = np.array(sorted(val_set))
    test_arr = np.array(sorted(test_set))

    def in_sorted(arr, vals):
        idx = np.searchsorted(arr, vals)
        valid = idx < len(arr)
        result = np.zeros(len(vals), dtype=bool)
        result[valid] = arr[idx[valid]] == vals[valid]
        return result

    src_in_train = in_sorted(train_arr, src)
    tgt_in_train = in_sorted(train_arr, tgt)
    src_in_val = in_sorted(val_arr, src)
    tgt_in_val = in_sorted(val_arr, tgt)
    src_in_test = in_sorted(test_arr, src)
    tgt_in_test = in_sorted(test_arr, tgt)

    train_mask = src_in_train & tgt_in_train
    val_mask = (src_in_train | src_in_val) & (tgt_in_train | tgt_in_val) & ~train_mask
    test_mask = src_in_test | tgt_in_test

    result = {
        "train": edges_df[train_mask][["source", "target"]].values,
        "val": edges_df[val_mask][["source", "target"]].values,
        "test": edges_df[test_mask][["source", "target"]].values,
        "train_nodes": train_set,
        "val_nodes": val_set,
        "test_nodes": test_set,
    }
    del src_in_train, tgt_in_train, src_in_val, tgt_in_val, src_in_test, tgt_in_test
    del train_mask, val_mask, test_mask, train_arr, val_arr, test_arr
    gc.collect()
    return result
