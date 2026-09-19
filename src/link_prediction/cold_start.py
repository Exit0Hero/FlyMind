#!/usr/bin/env python3
"""FlyMind Link Prediction — Cold-start node evaluation (Parts A-E)."""

import gc
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import *
from src.link_prediction.data_prep import (
    load_neuron_table, load_raw_edges, aggregate_edges,
    build_feature_matrix, log_mem,
)
from src.link_prediction.negative_sampling import (
    build_edge_index, sample_negatives,
    build_sorted_pairs, verify_not_in_edges,
)
from src.link_prediction.features import build_pair_features, build_edge_dicts, compute_heuristics
from src.link_prediction.evaluate import compute_all_metrics


def create_cold_start_node_split(edges_df, neuron_ids, seed=SEED):
    """Split neurons into train/val/test, then assign edges by node membership.

    A test candidate edge must involve at least one test-held-out node.
    Training graph must NOT contain edges involving test-held-out nodes.
    """
    np.random.seed(seed)
    unique_nodes = np.unique(neuron_ids)
    np.random.shuffle(unique_nodes)

    n = len(unique_nodes)
    n_train = int(n * COLD_TRAIN_RATIO)
    n_val = int(n * COLD_VAL_RATIO)

    train_nodes = unique_nodes[:n_train]
    val_nodes = unique_nodes[n_train:n_train + n_val]
    test_nodes = unique_nodes[n_train + n_val:]

    train_set = set(train_nodes.tolist())
    val_set = set(val_nodes.tolist())
    test_set = set(test_nodes.tolist())

    src = edges_df["source"].values
    tgt = edges_df["target"].values

    # Training edges: both endpoints in train_nodes
    train_mask = np.array([s in train_set and t in train_set for s, t in zip(src, tgt)])

    # Test edges: at least one endpoint in test_nodes, not in train-only
    test_mask = np.array([
        (s in test_set or t in test_set) and not (s in train_set and t in train_set)
        for s, t in zip(src, tgt)
    ])

    # Val edges: at least one endpoint in val_nodes, not in train-only, not in test
    val_mask = np.array([
        (s in val_set or t in val_set)
        and not (s in train_set and t in train_set)
        and not (s in test_set or t in test_set)
        for s, t in zip(src, tgt)
    ])

    return {
        "train": edges_df[train_mask][["source", "target"]].values,
        "val": edges_df[val_mask][["source", "target"]].values,
        "test": edges_df[test_mask][["source", "target"]].values,
        "train_nodes": train_set,
        "val_nodes": val_set,
        "test_nodes": test_set,
        "train_node_ids": train_nodes,
        "val_node_ids": val_nodes,
        "test_node_ids": test_nodes,
    }


def build_cold_start_candidates(test_edges, test_neg, train_edges, test_nodes, train_nodes):
    """Build candidate pairs for cold-start evaluation.

    Returns arrays for three direction categories:
    1. src=test, tgt=train (source test node)
    2. src=train, tgt=test (target test node)
    3. src=test, tgt=test (both test nodes)
    """
    test_pos = test_edges

    # Categorize test positives by direction
    mask_st = np.array([int(s) in test_nodes and int(t) in train_nodes for s, t in test_pos])
    mask_ts = np.array([int(s) in train_nodes and int(t) in test_nodes for s, t in test_pos])
    mask_tt = np.array([int(s) in test_nodes and int(t) in test_nodes for s, t in test_pos])

    return {
        "test_pos": test_pos,
        "test_neg": test_neg,
        "test_pos_st": test_pos[mask_st],
        "test_pos_ts": test_pos[mask_ts],
        "test_pos_tt": test_pos[mask_tt],
        "n_st": int(mask_st.sum()),
        "n_ts": int(mask_ts.sum()),
        "n_tt": int(mask_tt.sum()),
    }


def run_cold_start():
    """Run cold-start evaluation pipeline."""
    print("=" * 70)
    print("FLYMIND LINK PREDICTION — COLD-START EVALUATION")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LP_COLD_DIR = LP_DIR / "cold_start"
    LP_COLD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    log_mem("start")
    print("\n[1] Loading data...")
    nt = load_neuron_table()
    raw = load_raw_edges()
    edges = aggregate_edges(raw)
    del raw; gc.collect()
    print(f"  Neurons: {len(nt)}, Edges: {len(edges)}")
    log_mem("after data load")

    # ---- Build features ----
    print("\n[2] Building features...")
    X, root_ids, id_to_idx, cat_encoders, feature_cols = build_feature_matrix(
        nt, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    # Fill NaN/Inf for GNN compatibility
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    del nt; gc.collect()
    log_mem("after features")

    # ---- Cold-start split ----
    print("\n[3] Creating cold-start node split...")
    cold = create_cold_start_node_split(edges, root_ids, seed=SEED)
    print(f"  Train nodes: {len(cold['train_nodes'])}")
    print(f"  Val nodes: {len(cold['val_nodes'])}")
    print(f"  Test nodes: {len(cold['test_nodes'])}")
    print(f"  Train edges: {len(cold['train'])}")
    print(f"  Val edges: {len(cold['val'])}")
    print(f"  Test edges: {len(cold['test'])}")

    # Save node splits
    np.save(LP_COLD_DIR / "train_node_ids.npy", cold["train_node_ids"])
    np.save(LP_COLD_DIR / "val_node_ids.npy", cold["val_node_ids"])
    np.save(LP_COLD_DIR / "test_node_ids.npy", cold["test_node_ids"])

    # ---- Build training-only edge index ----
    print("\n[4] Building training-only edge index...")
    train_edge_index = build_edge_index(cold["train"])
    log_mem("after train edge index")

    # ---- Verify no leakage ----
    print("\n[5] Verifying no leakage...")
    train_set = cold["train_nodes"]
    test_set = cold["test_nodes"]
    for s, t in cold["train"]:
        assert int(s) in train_set and int(t) in train_set, \
            f"LEAKAGE: train edge ({s},{t}) touches non-train node!"
    print("  PASS: no training edge touches test/val nodes")

    for s, t in cold["test"]:
        assert int(s) in test_set or int(t) in test_set, \
            f"LEAKAGE: test edge ({s},{t}) has no test node!"
    print("  PASS: all test edges involve at least one test node")

    # ---- Sample negatives ----
    print("\n[6] Sampling cold-start negatives...")
    all_node_ids = np.unique(np.concatenate([cold["train"][:, 0], cold["train"][:, 1],
                                              cold["val"][:, 0], cold["val"][:, 1],
                                              cold["test"][:, 0], cold["test"][:, 1]]))

    # Negatives must not be in ANY positive edge (train, val, or test)
    # Use hash-based index for fast filtering, then collision-free verification
    full_pos_index = build_edge_index(edges[["source", "target"]].values)
    sorted_pos_pairs = build_sorted_pairs(edges["source"].values, edges["target"].values)

    # Sample extra to account for collision-filtered removals
    test_neg_raw = sample_negatives(all_node_ids, len(cold["test"]) + 10000, full_pos_index, seed=SEED + 10)
    val_neg_raw = sample_negatives(all_node_ids, len(cold["val"]) + 10000, full_pos_index, seed=SEED + 11)

    # Collision-free verification: remove any that are actual positives
    keep = verify_not_in_edges(test_neg_raw[:, 0], test_neg_raw[:, 1], sorted_pos_pairs)
    test_neg = test_neg_raw[keep][:len(cold["test"])]
    keep = verify_not_in_edges(val_neg_raw[:, 0], val_neg_raw[:, 1], sorted_pos_pairs)
    val_neg = val_neg_raw[keep][:len(cold["val"])]

    print(f"  Test neg: {len(test_neg)}, Val neg: {len(val_neg)}")

    # Final verification
    neg_leak = verify_not_in_edges(test_neg[:, 0], test_neg[:, 1], sorted_pos_pairs)
    n_leak = (~neg_leak).sum()
    assert n_leak == 0, f"LEAKAGE: {n_leak} test negatives are known positives!"
    print("  PASS: no test negatives are known positives")

    # ---- Build candidates ----
    candidates = build_cold_start_candidates(
        cold["test"], test_neg, cold["train"], test_set, train_set
    )
    print(f"\n  Test positive breakdown:")
    print(f"    src=test -> tgt=train: {candidates['n_st']}")
    print(f"    src=train -> tgt=test: {candidates['n_ts']}")
    print(f"    src=test -> tgt=test: {candidates['n_tt']}")
    print(f"    Total test positives: {len(candidates['test_pos'])}")

    # ---- Train-only graph heuristics ----
    print("\n[7] Computing train-only graph heuristics...")
    out_edges, in_edges = build_edge_dicts(cold["train"])

    test_all_pairs = np.vstack([cold["test"], test_neg])
    test_labels = np.concatenate([
        np.ones(len(cold["test"]), dtype=np.float32),
        np.zeros(len(test_neg), dtype=np.float32),
    ])

    t0 = time.time()
    heuristics = compute_heuristics(test_all_pairs, out_edges, in_edges)
    print(f"  Heuristics: {time.time()-t0:.1f}s")
    log_mem("after heuristics")

    # Save heuristics
    np.save(LP_COLD_DIR / "test_heuristics.npy", heuristics)
    np.save(LP_COLD_DIR / "test_labels.npy", test_labels)

    # ---- Random baseline ----
    print("\n--- Cold-start baselines ---")
    np.random.seed(SEED)
    random_scores = np.random.uniform(0, 1, len(test_labels))
    random_m = compute_all_metrics(test_labels, random_scores, "Cold:Random")

    # ---- Heuristic metrics ----
    hnames = ["common_out", "common_in", "total_common", "jaccard_out", "jaccard_in", "pa_out", "pa_in"]
    heur_metrics = {}
    for i, name in enumerate(hnames):
        scores = heuristics[:, i]
        s_min, s_max = scores.min(), scores.max()
        scores_norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else scores
        heur_metrics[name] = compute_all_metrics(test_labels, scores_norm, f"Cold:H:{name}")

    # ---- RF: node features only ----
    print("\n[8] Cold-start RF — node features only...")
    np.random.seed(SEED)
    n_tr = len(cold["train"])
    n_neg = len(test_neg)
    MAX_RF = 200_000
    n_pos_s = min(n_tr, MAX_RF // 2)
    n_neg_s = min(n_neg, MAX_RF // 2)
    pos_idx = np.random.choice(n_tr, n_pos_s, replace=False)
    neg_idx = np.random.choice(n_neg, n_neg_s, replace=False)

    rf_train_pairs = np.vstack([cold["train"][pos_idx], test_neg[neg_idx]])
    rf_train_labels = np.concatenate([
        np.ones(n_pos_s, dtype=np.float32),
        np.zeros(n_neg_s, dtype=np.float32),
    ])

    X_train_pair = build_pair_features(X, rf_train_pairs, id_to_idx)
    X_test_pair = build_pair_features(X, test_all_pairs, id_to_idx)
    del rf_train_pairs; gc.collect()
    log_mem("after cold pair features")

    from src.link_prediction.models import train_rf
    import pickle

    t0 = time.time()
    rf = train_rf(X_train_pair, rf_train_labels, n_estimators=100)
    print(f"  RF trained: {time.time()-t0:.1f}s")

    rf_scores = rf.predict_proba(X_test_pair)[:, 1]
    rf_node_m = compute_all_metrics(test_labels, rf_scores, "Cold:RF-node")

    del X_train_pair; gc.collect()

    # ---- RF: node features + train-only heuristics ----
    print("\n[9] Cold-start RF — node + train-only heuristics...")
    t0 = time.time()
    rf_train_pairs2 = np.vstack([cold["train"][pos_idx], test_neg[neg_idx]])
    rf_heur_train = compute_heuristics(rf_train_pairs2, out_edges, in_edges)
    rf_combo_train = np.hstack([build_pair_features(X, rf_train_pairs2, id_to_idx), rf_heur_train])
    rf_combo_test = np.hstack([X_test_pair, heuristics])

    rf_combo = train_rf(rf_combo_train, rf_train_labels, n_estimators=100)
    rf_combo_scores = rf_combo.predict_proba(rf_combo_test)[:, 1]
    rf_combo_m = compute_all_metrics(test_labels, rf_combo_scores, "Cold:RF-node+heur")
    print(f"  RF+heur: {time.time()-t0:.1f}s")

    del rf_combo_train, rf_combo_test, X_test_pair; gc.collect()
    log_mem("after cold RF")

    # ---- Direction breakdown ----
    print("\n[10] Direction breakdown...")
    direction_results = {}
    for direction, n_pairs in [("src=test->tgt=train", candidates["n_st"]),
                                ("src=train->tgt=test", candidates["n_ts"]),
                                ("src=test->tgt=test", candidates["n_tt"])]:
        if n_pairs > 0:
            print(f"  {direction}: {n_pairs} positive pairs")
            direction_results[direction] = {"n_positive": n_pairs}

    # ---- Save results ----
    results = {
        "split_sizes": {
            "train_nodes": len(cold["train_nodes"]),
            "val_nodes": len(cold["val_nodes"]),
            "test_nodes": len(cold["test_nodes"]),
            "train_edges": len(cold["train"]),
            "val_edges": len(cold["val"]),
            "test_edges": len(cold["test"]),
        },
        "direction_breakdown": {
            "src_test_tgt_train": candidates["n_st"],
            "src_train_tgt_test": candidates["n_ts"],
            "src_test_tgt_test": candidates["n_tt"],
        },
        "negative_samples": {
            "test_neg": len(test_neg),
            "val_neg": len(val_neg),
        },
        "random": random_m,
        "heuristics": heur_metrics,
        "rf_node_only": rf_node_m,
        "rf_node_plus_heuristic": rf_combo_m,
    }

    with open(REPORTS_DIR / "link_prediction_cold_start_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save splits for GNN
    for key in ["train", "val", "test"]:
        np.save(LP_COLD_DIR / f"split_{key}.npy", cold[key])
    np.save(LP_COLD_DIR / "test_neg.npy", test_neg)
    np.save(LP_COLD_DIR / "val_neg.npy", val_neg)
    np.save(LP_COLD_DIR / "X_features.npy", X)
    json.dump({str(k): v for k, v in id_to_idx.items()},
              open(LP_COLD_DIR / "id_to_idx.json", "w"))

    print("\n" + "=" * 70)
    print("COLD-START PARTS A-E COMPLETE")
    print("=" * 70)
    log_mem("final")

    return results


if __name__ == "__main__":
    run_cold_start()
