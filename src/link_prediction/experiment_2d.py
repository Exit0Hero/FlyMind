#!/usr/bin/env python3
"""Experiment 2D: Explainable Candidate Connection Ranking.

Uses the validated RF node-feature model to rank plausible candidate directed
connections between neurons. All outputs are model-suggested hypotheses, NOT
biological discoveries.

Memory target: < 2 GB peak RSS.
"""

import gc
import json
import os
import pickle
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.link_prediction.config import (
    LP_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR, SEED,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------
def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def log_mem(tag=""):
    rss = _rss_mb()
    print(f"  [MEM:{tag}] RSS: {rss:.0f} MB")
    return rss


def check_mem(tag="", limit_mb=2500):
    rss = _rss_mb()
    if rss > limit_mb:
        raise MemoryError(f"RSS {rss:.0f} MB exceeds {limit_mb} MB limit at {tag}")
    return rss


# ---------------------------------------------------------------------------
# Pair feature construction (chunked)
# ---------------------------------------------------------------------------
def build_pair_features_chunked(X, src_ids, dst_ids, chunk_size=10000):
    """Yield pair feature chunks: [x_A, x_B, |x_A-x_B|, x_A*x_B]."""
    n = len(src_ids)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ia = src_ids[start:end]
        ib = dst_ids[start:end]
        xa = X[ia]
        xb = X[ib]
        chunk = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
        yield chunk, start, end
        del xa, xb


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def generate_candidates(
    all_node_ids, sorted_edges, n_sources=10000, targets_per_source=100,
    seed=42, chunk_size=10000,
):
    """Generate candidate (source, target) pairs, excluding known edges and self-loops.

    Yields chunks of (src_ids, dst_ids) as raw node IDs.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(all_node_ids)

    # Select source neurons (random subset)
    source_indices = rng.choice(n_nodes, size=min(n_sources, n_nodes), replace=False)
    source_ids = all_node_ids[source_indices]

    # For each source, sample candidate targets
    all_src = []
    all_dst = []

    for src_id in source_ids:
        # Sample candidate targets
        cand_indices = rng.choice(n_nodes, size=targets_per_source * 3, replace=False)
        cand_ids = all_node_ids[cand_indices]

        # Exclude self-loops
        valid = cand_ids != src_id
        cand_ids = cand_ids[valid]

        # Exclude known edges (binary search on sorted pairs)
        keep = verify_not_in_edges(
            np.full(len(cand_ids), src_id, dtype=np.int64),
            cand_ids,
            sorted_edges,
        )
        cand_ids = cand_ids[keep]

        # Take up to targets_per_source
        n_take = min(len(cand_ids), targets_per_source)
        if n_take > 0:
            chosen = cand_ids[:n_take]
            all_src.append(np.full(n_take, src_id, dtype=np.int64))
            all_dst.append(chosen)

        # Yield in chunks to control memory
        if len(all_src) >= chunk_size // targets_per_source:
            src_arr = np.concatenate(all_src)
            dst_arr = np.concatenate(all_dst)
            yield src_arr, dst_arr
            all_src = []
            all_dst = []
            gc.collect()

    # Final chunk
    if all_src:
        src_arr = np.concatenate(all_src)
        dst_arr = np.concatenate(all_dst)
        yield src_arr, dst_arr


# ---------------------------------------------------------------------------
# Validation against held-out cold-start edges
# ---------------------------------------------------------------------------
def validate_on_held_out(rf, X, id_to_idx, cs_test_pos, cs_test_neg, chunk_size=10000):
    """Evaluate RF on cold-start held-out positives vs negatives."""
    print("\n--- Validation on held-out cold-start edges ---")
    n_pos = len(cs_test_pos)
    n_neg = len(cs_test_neg)

    # Build index arrays
    pos_src = np.array([id_to_idx[int(i)] for i in cs_test_pos[:, 0]], dtype=np.int64)
    pos_dst = np.array([id_to_idx[int(i)] for i in cs_test_pos[:, 1]], dtype=np.int64)
    neg_src = np.array([id_to_idx[int(i)] for i in cs_test_neg[:, 0]], dtype=np.int64)
    neg_dst = np.array([id_to_idx[int(i)] for i in cs_test_neg[:, 1]], dtype=np.int64)

    all_src = np.concatenate([pos_src, neg_src])
    all_dst = np.concatenate([pos_dst, neg_dst])
    labels = np.concatenate([
        np.ones(n_pos, dtype=np.float32),
        np.zeros(n_neg, dtype=np.float32),
    ])
    n_total = n_pos + n_neg

    # Score in chunks
    scores = np.empty(n_total, dtype=np.float32)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_features, _, _ = next(
            build_pair_features_chunked(X, all_src[start:end], all_dst[start:end])
        )
        scores[start:end] = rf.predict_proba(chunk_features)[:, 1]
        del chunk_features

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))

    # Precision@K and Recall@K
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    results = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    n_pos_int = int(n_pos)
    for k in [10, 50, 100]:
        if k <= len(scores):
            top_k = sorted_labels[:k]
            results[f"precision_at_{k}"] = float(top_k.sum() / k)
            results[f"recall_at_{k}"] = float(top_k.sum() / max(n_pos_int, 1))
            results[f"hits_at_{k}"] = int(top_k.sum())

    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    for k in [10, 50, 100]:
        pk = f"precision_at_{k}"
        rk = f"recall_at_{k}"
        if pk in results:
            print(f"  P@{k}: {results[pk]:.4f} | R@{k}: {results[rk]:.4f} | Hits@{k}: {results[f'hits_at_{k}']}")

    del all_src, all_dst, labels, scores, sorted_indices, sorted_labels
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Score distribution statistics
# ---------------------------------------------------------------------------
def compute_score_stats(scores):
    """Compute summary statistics over a score array."""
    return {
        "count": len(scores),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


# ---------------------------------------------------------------------------
# Feature importance analysis
# ---------------------------------------------------------------------------
def compute_feature_importance(rf, feature_cols):
    """Return sorted feature importances."""
    importances = rf.feature_importances_
    pairs = list(zip(feature_cols, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def group_feature_importance(feature_importance):
    """Group importances into source/target/diff/interaction categories."""
    groups = {
        "source biology": 0.0,
        "target biology": 0.0,
        "source-target difference": 0.0,
        "source-target interaction": 0.0,
    }
    n_feat = len(feature_importance) // 4
    for i, (name, imp) in enumerate(feature_importance):
        if i < n_feat:
            groups["source biology"] += imp
        elif i < 2 * n_feat:
            groups["target biology"] += imp
        elif i < 3 * n_feat:
            groups["source-target difference"] += imp
        else:
            groups["source-target interaction"] += imp
    return groups


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment_2d(n_sources=10000, targets_per_source=100, smoke_test=False):
    print("=" * 70)
    print("EXPERIMENT 2D: Explainable Candidate Connection Ranking")
    print("=" * 70)
    log_mem("program start")

    # ── Stage 1: Load data ──
    print("\n[1] Loading data...")
    data_dir = str(LP_DIR)
    cold_dir = str(LP_DIR / "cold_start")

    # Load RF model
    with open(MODELS_DIR / "link_prediction_rf.pkl", "rb") as f:
        rf_data = pickle.load(f)
    rf = rf_data["model"]
    feature_cols = rf_data["feature_cols"]
    log_mem("after loading RF model")

    # Load node features
    X = np.load(os.path.join(data_dir, "X_features.npy"))
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert X.dtype == np.float32
    with open(os.path.join(data_dir, "id_to_idx.json")) as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    all_node_ids = np.array(sorted(id_to_idx.keys()), dtype=np.int64)
    log_mem("after loading features")

    # Load edges for known-edge filtering
    edges_df = pd.read_parquet(os.path.join(data_dir, "edges_aggregated.parquet"))
    all_edges = edges_df[["source", "target"]].values.astype(np.int64)
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    del edges_df
    gc.collect()
    log_mem("after loading edges")

    # Load cold-start test data for validation
    cs_test_pos = np.load(os.path.join(cold_dir, "split_test.npy"))
    cs_test_neg = np.load(os.path.join(cold_dir, "test_neg.npy"))
    log_mem("after loading cold-start test")

    print(f"  Nodes: {len(all_node_ids)}, Features: {X.shape[1]}")
    print(f"  Edges: {len(all_edges)}, RF estimators: {rf.n_estimators}")
    print(f"  Cold-start test: {len(cs_test_pos)} pos, {len(cs_test_neg)} neg")

    # ── Stage 2: Validation on held-out edges ──
    print("\n[2] Validation on held-out cold-start edges...")
    val_results = validate_on_held_out(rf, X, id_to_idx, cs_test_pos, cs_test_neg)
    log_mem("after validation")

    # ── Stage 3: Candidate generation and scoring ──
    if smoke_test:
        n_sources = 100
        targets_per_source = 100
        print(f"\n[3] SMOKE TEST: {n_sources} sources × {targets_per_source} targets...")

    print(f"\n[3] Generating candidates ({n_sources} sources × {targets_per_source} targets)...")
    check_mem("before candidate generation")

    # Use a bounded top-K structure per source
    # For each source: list of (score, target_id) tuples, keep top 10
    top_k_per_source = defaultdict(list)
    all_scores = []
    n_candidates_total = 0
    n_chunks = 0

    for src_chunk, dst_chunk in generate_candidates(
        all_node_ids, sorted_edges,
        n_sources=n_sources, targets_per_source=targets_per_source,
        seed=SEED,
    ):
        n_candidates_total += len(src_chunk)
        n_chunks += 1

        # Build pair features and score in sub-chunks
        for start in range(0, len(src_chunk), 5000):
            end = min(start + 5000, len(src_chunk))
            src_ids_sub = src_chunk[start:end]
            dst_ids_sub = dst_chunk[start:end]

            # Map raw node IDs to feature matrix indices
            src_idx = np.array([id_to_idx[int(s)] for s in src_ids_sub], dtype=np.int64)
            dst_idx = np.array([id_to_idx[int(d)] for d in dst_ids_sub], dtype=np.int64)

            # Build pair features
            xa = X[src_idx]
            xb = X[dst_idx]
            pair_features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
            del xa, xb

            # Score
            scores = rf.predict_proba(pair_features)[:, 1]
            del pair_features

            # Track scores for distribution
            all_scores.append(scores.copy())

            # Update top-K per source
            for i in range(len(src_ids_sub)):
                src_id = int(src_ids_sub[i])
                dst_id = int(dst_ids_sub[i])
                score = float(scores[i])
                heap = top_k_per_source[src_id]
                if len(heap) < 10:
                    heap.append((score, dst_id))
                    heap.sort(key=lambda x: x[0], reverse=True)
                elif score > heap[-1][0]:
                    heap[-1] = (score, dst_id)
                    heap.sort(key=lambda x: x[0], reverse=True)

            del scores, src_idx, dst_idx, src_ids_sub, dst_ids_sub

        del src_chunk, dst_chunk
        gc.collect()

        if n_chunks % 5 == 0:
            log_mem(f"after chunk {n_chunks}")

    log_mem("after all candidates")

    # Combine all scores
    all_scores_arr = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    del all_scores
    gc.collect()

    score_stats = compute_score_stats(all_scores_arr)
    print(f"\n  Total candidates scored: {n_candidates_total}")
    print(f"  Sources with candidates: {len(top_k_per_source)}")
    print(f"  Score distribution:")
    for k, v in score_stats.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Stage 4: Build final top candidates table ──
    print("\n[4] Building final candidates table...")
    rows = []
    for src_id, heap in top_k_per_source.items():
        for rank, (score, dst_id) in enumerate(heap, 1):
            rows.append({
                "source_root_id": src_id,
                "target_root_id": dst_id,
                "predicted_score": score,
                "rank": rank,
            })

    candidates_df = pd.DataFrame(rows)
    candidates_df.sort_values("predicted_score", ascending=False, inplace=True)
    candidates_df.reset_index(drop=True, inplace=True)
    print(f"  Final candidates: {len(candidates_df)}")

    # ── Stage 5: Post-hoc biological analysis ──
    print("\n[5] Post-hoc biological analysis...")
    nt = pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")

    # Build lookup dicts for post-hoc features
    nt_lookup = nt.set_index("root_id")[[
        "super_class", "primary_type", "nt_type", "flow",
        "coord_x", "coord_y", "coord_z", "length_nm",
    ]].to_dict("index")

    # Enrich top 20 candidates
    top20 = candidates_df.head(20).copy()
    for col in ["source_super_class", "target_super_class",
                "source_primary_type", "target_primary_type",
                "source_nt_type", "target_nt_type",
                "source_flow", "target_flow",
                "source_coord_x", "source_coord_y", "source_coord_z",
                "target_coord_x", "target_coord_y", "target_coord_z"]:
        top20[col] = None

    for idx, row in top20.iterrows():
        src_info = nt_lookup.get(int(row["source_root_id"]), {})
        dst_info = nt_lookup.get(int(row["target_root_id"]), {})
        top20.at[idx, "source_super_class"] = src_info.get("super_class", "N/A") or "N/A"
        top20.at[idx, "target_super_class"] = dst_info.get("super_class", "N/A") or "N/A"
        top20.at[idx, "source_primary_type"] = src_info.get("primary_type", "N/A") or "N/A"
        top20.at[idx, "target_primary_type"] = dst_info.get("primary_type", "N/A") or "N/A"
        top20.at[idx, "source_nt_type"] = src_info.get("nt_type", "N/A") or "N/A"
        top20.at[idx, "target_nt_type"] = dst_info.get("nt_type", "N/A") or "N/A"
        top20.at[idx, "source_flow"] = src_info.get("flow", "N/A") or "N/A"
        top20.at[idx, "target_flow"] = dst_info.get("flow", "N/A") or "N/A"
        top20.at[idx, "source_coord_x"] = src_info.get("coord_x", None)
        top20.at[idx, "source_coord_y"] = src_info.get("coord_y", None)
        top20.at[idx, "source_coord_z"] = src_info.get("coord_z", None)
        top20.at[idx, "target_coord_x"] = dst_info.get("coord_x", None)
        top20.at[idx, "target_coord_y"] = dst_info.get("coord_y", None)
        top20.at[idx, "target_coord_z"] = dst_info.get("coord_z", None)

    del nt
    gc.collect()

    # ── Stage 6: Feature importance ──
    print("\n[6] Feature importance analysis...")
    feat_imp = compute_feature_importance(rf, feature_cols)
    group_imp = group_feature_importance(feat_imp)

    print("  Top 10 features:")
    for name, imp in feat_imp[:10]:
        print(f"    {name:30s} {imp:.4f}")
    print("  Group importances:")
    for group, imp in group_imp.items():
        print(f"    {group:30s} {imp:.4f}")

    # ── Stage 7: Connection type analysis ──
    print("\n[7] Connection type analysis...")
    # Reload neuron table for type analysis (lightweight)
    nt_types = pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")[["root_id", "nt_type"]].set_index("root_id")["nt_type"].to_dict()
    top50 = candidates_df.head(50).copy()
    nt_type_counts = defaultdict(int)
    for idx, row in top50.iterrows():
        src_nt = nt_types.get(int(row["source_root_id"]), None)
        dst_nt = nt_types.get(int(row["target_root_id"]), None)
        src_nt = src_nt if pd.notna(src_nt) else "unknown"
        dst_nt = dst_nt if pd.notna(dst_nt) else "unknown"
        key = f"{src_nt} -> {dst_nt}"
        nt_type_counts[key] += 1
    print("  Top NT type pairs in top-50 candidates:")
    for pair, count in sorted(nt_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair}: {count}")
    del nt_types
    gc.collect()

    # ── Stage 8: Save outputs ──
    print("\n[8] Saving outputs...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "results" / "candidates").mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "results" / "figures" / "experiment_2d").mkdir(parents=True, exist_ok=True)

    # Save candidates CSV
    csv_path = Path.cwd() / "results" / "candidates" / "top_candidate_connections.csv"
    candidates_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save top 20 with biological annotations
    top20_path = Path.cwd() / "results" / "candidates" / "top20_annotated.csv"
    top20.to_csv(top20_path, index=False)
    print(f"  Saved: {top20_path}")

    # Save JSON results
    all_results = {
        "validation": val_results,
        "candidate_generation": {
            "n_sources": n_sources,
            "targets_per_source": targets_per_source,
            "total_candidates_scored": n_candidates_total,
            "sources_with_candidates": len(top_k_per_source),
            "final_top_candidates": len(candidates_df),
        },
        "score_distribution": score_stats,
        "feature_importance_top10": [(n, float(v)) for n, v in feat_imp[:10]],
        "feature_importance_groups": {k: float(v) for k, v in group_imp.items()},
    }
    json_path = REPORTS_DIR / "experiment_2d_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {json_path}")

    log_mem("final")
    print("\nDone!")
    return all_results, candidates_df, top20


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (100 sources)")
    parser.add_argument("--n-sources", type=int, default=10000)
    parser.add_argument("--targets-per-source", type=int, default=100)
    args = parser.parse_args()

    run_experiment_2d(
        n_sources=args.n_sources,
        targets_per_source=args.targets_per_source,
        smoke_test=args.smoke,
    )
