#!/usr/bin/env python3
"""FlyMind Link Prediction — Part 1: data, heuristics, RF (memory-safe)."""

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
    build_feature_matrix, create_random_edge_split, create_cold_start_split,
    log_mem,
)
from src.link_prediction.negative_sampling import (
    build_edge_index, sample_negatives, sample_hard_negatives,
)
from src.link_prediction.features import (
    build_pair_features, build_edge_dicts, compute_heuristics,
)
from src.link_prediction.evaluate import compute_all_metrics


def run_heuristics_and_rf():
    """Run heuristic baselines and Random Forest."""
    print("=" * 70)
    print("FLYMIND LINK PREDICTION — PHASES 1-5 (memory-safe)")
    print("=" * 70)

    LP_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Load data ---
    log_mem("start")
    print("\n[1] Loading neuron table...")
    nt = load_neuron_table()
    print(f"  Neurons: {len(nt)}")
    log_mem("after neuron table")

    print("\n[2] Loading and aggregating edges...")
    raw = load_raw_edges()
    print(f"  Raw edges: {len(raw)}")
    edges = aggregate_edges(raw)
    print(f"  Aggregated edges: {len(edges)}")
    edges.to_parquet(LP_DIR / "edges_aggregated.parquet", index=False)
    del raw; gc.collect()
    log_mem("after edge aggregation")

    # --- Phase 2: Features ---
    print("\n[3] Building feature matrix...")
    X, root_ids, id_to_idx, cat_encoders, feature_cols = build_feature_matrix(
        nt, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    print(f"  Features: {X.shape[1]}")
    del nt; gc.collect()
    log_mem("after features")

    # --- Phase 3: Edge split ---
    print("\n[4] Creating random edge split...")
    all_pos = edges[["source", "target"]].values
    all_node_ids = np.unique(np.concatenate([all_pos[:, 0], all_pos[:, 1]]))
    print(f"  Total unique nodes: {len(all_node_ids)}")

    split = create_random_edge_split(edges)
    print(f"  Train: {len(split['train'])} | Val: {len(split['val'])} | Test: {len(split['test'])}")

    # Compact edge index (~30 MB vs ~450 MB for tuple set)
    print("  Building compact edge index...")
    all_edge_index = build_edge_index(all_pos)
    del all_pos; gc.collect()
    log_mem("after edge index")

    # --- Phase 4: Negative sampling ---
    print("\n[5] Sampling negatives (memory-safe)...")
    t0 = time.time()
    train_neg = sample_negatives(all_node_ids, len(split["train"]), all_edge_index, seed=SEED)
    val_neg = sample_negatives(all_node_ids, len(split["val"]), all_edge_index, seed=SEED + 1)
    test_neg = sample_negatives(all_node_ids, len(split["test"]), all_edge_index, seed=SEED + 2)
    print(f"  Train neg: {len(train_neg)} | Val neg: {len(val_neg)} | Test neg: {len(test_neg)}")
    print(f"  Neg sampling: {time.time()-t0:.1f}s")
    log_mem("after random negatives")

    print("\n[6] Sampling hard negatives...")
    train_edge_index = build_edge_index(split["train"])
    test_neg_hard = sample_hard_negatives(
        split["test"], all_node_ids, train_edge_index,
        len(split["test"]) * 5, split["train"], seed=SEED + 3,
    )
    print(f"  Hard neg: {len(test_neg_hard)}")
    del train_edge_index; gc.collect()
    log_mem("after hard negatives")

    # --- Phase 5: Graph heuristics (test set only) ---
    print("\n[7] Computing graph heuristics on test set...")
    out_edges, in_edges = build_edge_dicts(split["train"])

    test_all_pairs = np.vstack([split["test"], test_neg])
    test_labels = np.concatenate([
        np.ones(len(split["test"]), dtype=np.float32),
        np.zeros(len(test_neg), dtype=np.float32),
    ])

    t0 = time.time()
    heuristics = compute_heuristics(test_all_pairs, out_edges, in_edges)
    print(f"  Heuristics: {time.time()-t0:.1f}s")
    log_mem("after test heuristics")

    # Random baseline
    print("\n--- Baseline 1: Random ---")
    np.random.seed(SEED)
    random_scores = np.random.uniform(0, 1, len(test_labels))
    random_metrics = compute_all_metrics(test_labels, random_scores, "Random")

    # Heuristic metrics
    print("\n--- Baselines: Graph heuristics ---")
    heuristic_names = ["common_out", "common_in", "total_common", "jaccard_out", "jaccard_in", "pa_out", "pa_in"]
    heuristic_metrics = {}
    for i, name in enumerate(heuristic_names):
        scores = heuristics[:, i]
        s_min, s_max = scores.min(), scores.max()
        scores_norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else scores
        heuristic_metrics[name] = compute_all_metrics(test_labels, scores_norm, f"H:{name}")

    # Save test heuristics (needed for part 2)
    np.save(LP_DIR / "test_heuristics.npy", heuristics)
    np.save(LP_DIR / "test_labels.npy", test_labels)
    del heuristics; gc.collect()
    log_mem("after heuristic metrics")

    # --- Phase 6: Random Forest ---
    print("\n[8] Random Forest — subsampled for speed...")
    np.random.seed(SEED)
    n_train_pos = len(split["train"])
    n_train_neg = len(train_neg)
    MAX_RF = 200_000
    n_pos_sample = min(n_train_pos, MAX_RF // 2)
    n_neg_sample = min(n_train_neg, MAX_RF // 2)
    pos_idx = np.random.choice(n_train_pos, n_pos_sample, replace=False)
    neg_idx = np.random.choice(n_train_neg, n_neg_sample, replace=False)

    rf_train_pairs = np.vstack([split["train"][pos_idx], train_neg[neg_idx]])
    rf_train_labels = np.concatenate([
        np.ones(n_pos_sample, dtype=np.float32),
        np.zeros(n_neg_sample, dtype=np.float32),
    ])

    print(f"  RF train: {len(rf_train_pairs)} pairs")

    from src.link_prediction.models import train_rf
    import pickle

    # Compute heuristics for RF train pairs BEFORE deleting rf_train_pairs
    print("  Computing heuristics for RF train pairs...")
    rf_heur_train = compute_heuristics(rf_train_pairs, out_edges, in_edges)

    X_train_pair = build_pair_features(X, rf_train_pairs, id_to_idx)
    X_test_pair = build_pair_features(X, test_all_pairs, id_to_idx)
    del rf_train_pairs; gc.collect()
    log_mem("after pair features")

    t0 = time.time()
    rf = train_rf(X_train_pair, rf_train_labels, n_estimators=100)
    print(f"  RF trained: {time.time()-t0:.1f}s")

    with open(MODELS_DIR / "link_prediction_rf.pkl", "wb") as f:
        pickle.dump({"model": rf, "feature_cols": feature_cols}, f)

    rf_scores = rf.predict_proba(X_test_pair)[:, 1]
    rf_metrics = compute_all_metrics(test_labels, rf_scores, "RF node-only")

    # RF + heuristics
    print("\n[9] RF — node features + graph heuristics...")
    t0 = time.time()
    rf_train_combo = np.hstack([X_train_pair, rf_heur_train])
    del X_train_pair, rf_heur_train; gc.collect()
    rf_test_combo = np.hstack([X_test_pair, np.load(LP_DIR / "test_heuristics.npy")])

    rf_combo = train_rf(rf_train_combo, rf_train_labels, n_estimators=100)
    rf_combo_scores = rf_combo.predict_proba(rf_test_combo)[:, 1]
    rf_combo_metrics = compute_all_metrics(test_labels, rf_combo_scores, "RF node+heuristic")
    print(f"  RF+heur trained: {time.time()-t0:.1f}s")

    del rf_train_combo, rf_test_combo, X_test_pair
    gc.collect()
    log_mem("after RF")

    # --- Save results ---
    all_results = {
        "random": random_metrics,
        "heuristics": heuristic_metrics,
        "rf_node_only": rf_metrics,
        "rf_node_plus_heuristic": rf_combo_metrics,
    }
    with open(REPORTS_DIR / "link_prediction_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save intermediates for part 2 (compact)
    np.save(LP_DIR / "X_features.npy", X)
    np.save(LP_DIR / "train_neg.npy", train_neg)
    np.save(LP_DIR / "val_neg.npy", val_neg)
    np.save(LP_DIR / "test_neg.npy", test_neg)
    np.save(LP_DIR / "test_neg_hard.npy", test_neg_hard)

    # Save split (as dict of arrays)
    for key in ["train", "val", "test"]:
        np.save(LP_DIR / f"split_{key}.npy", split[key])

    np.save(LP_DIR / "all_node_ids.npy", all_node_ids)

    # Save id_to_idx mapping
    json.dump({str(k): v for k, v in id_to_idx.items()},
              open(LP_DIR / "id_to_idx.json", "w"))

    print("\n" + "=" * 70)
    print("PHASES 1-5 COMPLETE — Results saved.")
    print("=" * 70)
    log_mem("final")

    return all_results


if __name__ == "__main__":
    run_heuristics_and_rf()
