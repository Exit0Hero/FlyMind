#!/usr/bin/env python3
"""Experiment 3: Candidate Ranking Robustness, Calibration, and Biological Plausibility.

Evaluates whether the RF candidate-ranking system is:
1. genuinely discriminative (per-source ranking)
2. properly calibrated (Brier, log loss, calibration curve)
3. robust to candidate sampling (multi-seed stability)
4. not simply producing saturated 1.0 scores
5. producing scientifically interpretable candidate rankings

Memory target: < 2 GB peak RSS. Seeds processed sequentially with gc.collect().
"""

import gc
import json
import os
import pickle
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
from src.link_prediction.negative_sampling import (
    build_sorted_pairs, verify_not_in_edges, edge_exists,
)


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


# ---------------------------------------------------------------------------
# Pair feature construction (chunked)
# ---------------------------------------------------------------------------
def build_pair_features_chunked(X, src_ids, dst_ids, chunk_size=10000):
    """Yield (chunk_features, start, end) for pair scoring."""
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


def score_pairs(rf, X, src_idx, dst_idx, chunk_size=10000):
    """Score (src, dst) pairs using chunked inference. Returns score array."""
    n = len(src_idx)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xa = X[src_idx[start:end]]
        xb = X[dst_idx[start:end]]
        features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
        scores[start:end] = rf.predict_proba(features)[:, 1]
        del xa, xb, features
    return scores


# ---------------------------------------------------------------------------
# 3A: Per-source ranking evaluation
# ---------------------------------------------------------------------------
def per_source_ranking(rf, X, id_to_idx, cs_test_pos, cs_test_neg,
                       all_node_ids, sorted_edges, n_sources=2000,
                       n_neg_per_source=100, seed=42, chunk_size=5000):
    """Evaluate per-source Recall@K and Hit Rate@K.

    For each source with held-out positives:
    1. gather its held-out positive targets
    2. sample n_neg_per_source negative targets (not in training/test edges)
    3. score all (source, target) pairs
    4. rank targets, compute Recall@K and Hit Rate@K
    """
    print("  Per-source ranking evaluation...")
    rng = np.random.default_rng(seed)

    # Group test positives by source
    pos_by_source = defaultdict(list)
    for s, t in cs_test_pos:
        pos_by_source[int(s)].append(int(t))

    # Group test negatives by source (from test_neg which mirrors split_test)
    neg_by_source = defaultdict(list)
    for s, t in cs_test_neg:
        neg_by_source[int(s)].append(int(t))

    # Select sources to evaluate (subsample for speed)
    eligible_sources = [s for s, v in pos_by_source.items() if len(v) >= 1]
    rng.shuffle(eligible_sources)
    eval_sources = eligible_sources[:n_sources]

    # Build set of all known edges for this source (training + test)
    # Use training edges
    train_edges = np.load(str(LP_DIR / "cold_start" / "split_train.npy"))
    train_edges_by_source = defaultdict(set)
    for s, t in train_edges:
        train_edges_by_source[int(s)].add(int(t))
    del train_edges
    gc.collect()

    k_values = [10, 50, 100]
    recall_lists = {k: [] for k in k_values}
    hit_lists = {k: [] for k in k_values}
    eligible_count = 0
    total_positives = 0

    node_set = set(all_node_ids.tolist())

    for src_id in eval_sources:
        pos_targets = pos_by_source[src_id]
        n_pos = len(pos_targets)
        if n_pos == 0:
            continue
        total_positives += n_pos

        # Sample negatives for this source
        sampled_neg = set(neg_by_source.get(src_id, []))
        collected = 0
        neg_targets = []
        attempts = 0
        while collected < n_neg_per_source and attempts < n_neg_per_source * 20:
            attempts += 1
            cand = int(rng.choice(all_node_ids))
            if cand == src_id:
                continue
            if cand in sampled_neg or cand in train_edges_by_source.get(src_id, set()):
                continue
            if cand in pos_targets:
                continue
            neg_targets.append(cand)
            sampled_neg.add(cand)
            collected += 1

        if len(neg_targets) == 0:
            continue

        # Build candidate set: positives + negatives
        all_targets = np.array(pos_targets + neg_targets, dtype=np.int64)
        is_positive = np.array(
            [1] * len(pos_targets) + [0] * len(neg_targets), dtype=np.float32
        )

        # Map to feature indices
        src_arr = np.full(len(all_targets), src_id, dtype=np.int64)
        src_idx = np.array([id_to_idx[int(s)] for s in src_arr], dtype=np.int64)
        dst_idx = np.array([id_to_idx[int(d)] for d in all_targets], dtype=np.int64)

        scores = score_pairs(rf, X, src_idx, dst_idx, chunk_size=chunk_size)

        # Rank by score (descending)
        rank_order = np.argsort(-scores)
        sorted_labels = is_positive[rank_order]

        # Compute per-source Recall@K
        for k in k_values:
            if k <= len(sorted_labels):
                hits = float(sorted_labels[:k].sum())
                recall_lists[k].append(hits / max(n_pos, 1))
                hit_lists[k].append(1.0 if hits > 0 else 0.0)

        eligible_count += 1

        if eligible_count % 500 == 0:
            print(f"    Processed {eligible_count}/{n_sources} sources...")

    # Compute summary statistics
    results = {"n_sources_evaluated": eligible_count, "total_positives": total_positives}
    for k in k_values:
        r = np.array(recall_lists[k]) if recall_lists[k] else np.array([0.0])
        h = np.array(hit_lists[k]) if hit_lists[k] else np.array([0.0])
        results[f"mean_recall@{k}"] = float(np.mean(r))
        results[f"median_recall@{k}"] = float(np.median(r))
        results[f"hit_rate@{k}"] = float(np.mean(h))

    print(f"    Evaluated {eligible_count} sources ({total_positives} total positives)")
    for k in k_values:
        print(f"    Recall@{k}: mean={results[f'mean_recall@{k}']:.4f} "
              f"median={results[f'median_recall@{k}']:.4f} "
              f"hit_rate={results[f'hit_rate@{k}']:.4f}")

    del train_edges_by_source
    gc.collect()
    return results


def random_baseline_ranking(n_sources=2000, n_neg_per_source=100, n_pos_per_source=5, seed=42):
    """Compute ranking metrics for a random baseline.

    Uses random scores instead of RF predictions.
    """
    print("  Random baseline ranking...")
    rng = np.random.default_rng(seed)
    k_values = [10, 50, 100]
    recall_lists = {k: [] for k in k_values}
    hit_lists = {k: [] for k in k_values}

    for _ in range(n_sources):
        n_pos = n_pos_per_source
        n_neg = n_neg_per_source
        is_positive = np.array([1]*n_pos + [0]*n_neg, dtype=np.float32)
        random_scores = rng.random(n_pos + n_neg).astype(np.float32)
        rank_order = np.argsort(-random_scores)
        sorted_labels = is_positive[rank_order]

        for k in k_values:
            if k <= len(sorted_labels):
                hits = float(sorted_labels[:k].sum())
                recall_lists[k].append(hits / max(n_pos, 1))
                hit_lists[k].append(1.0 if hits > 0 else 0.0)

    results = {}
    for k in k_values:
        r = np.array(recall_lists[k])
        h = np.array(hit_lists[k])
        results[f"mean_recall@{k}"] = float(np.mean(r))
        results[f"median_recall@{k}"] = float(np.median(r))
        results[f"hit_rate@{k}"] = float(np.mean(h))

    for k in k_values:
        print(f"    Random Recall@{k}: mean={results[f'mean_recall@{k}']:.4f} "
              f"hit_rate={results[f'hit_rate@{k}']:.4f}")
    return results


# ---------------------------------------------------------------------------
# 3B: Candidate sampling robustness
# ---------------------------------------------------------------------------
def seed_stability_analysis(rf, X, id_to_idx, all_node_ids, sorted_edges,
                            n_sources=10000, targets_per_source=100, seeds=(43, 44, 45)):
    """Repeat candidate generation with different seeds, compare distributions."""
    print("  Seed stability analysis...")
    from src.link_prediction.experiment_2d import generate_candidates, compute_score_stats

    results = {}
    seed_top100 = {}

    for seed in seeds:
        print(f"    Seed {seed}...")
        all_scores = []
        top100_pairs = set()

        for src_chunk, dst_chunk in generate_candidates(
            all_node_ids, sorted_edges,
            n_sources=n_sources, targets_per_source=targets_per_source,
            seed=seed,
        ):
            for start in range(0, len(src_chunk), 5000):
                end = min(start + 5000, len(src_chunk))
                src_ids_sub = src_chunk[start:end]
                dst_ids_sub = dst_chunk[start:end]

                src_idx = np.array([id_to_idx[int(s)] for s in src_ids_sub], dtype=np.int64)
                dst_idx = np.array([id_to_idx[int(d)] for d in dst_ids_sub], dtype=np.int64)

                xa = X[src_idx]
                xb = X[dst_idx]
                pair_features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
                scores = rf.predict_proba(pair_features)[:, 1]
                all_scores.append(scores.copy())

                # Track top-100 by score
                for i in range(len(src_ids_sub)):
                    top100_pairs.add((float(scores[i]), int(src_ids_sub[i]), int(dst_ids_sub[i])))

                del xa, xb, pair_features, scores, src_idx, dst_idx
            del src_chunk, dst_chunk
            gc.collect()

        all_scores_arr = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
        del all_scores
        gc.collect()

        score_stats = compute_score_stats(all_scores_arr)
        n_exact_1 = int((all_scores_arr == 1.0).sum())
        n_ge_09 = int((all_scores_arr >= 0.9).sum())

        results[str(seed)] = {
            "score_stats": score_stats,
            "n_exact_1": n_exact_1,
            "n_ge_09": n_ge_09,
        }

        # Keep top-100 pairs (by score desc, then src, then dst)
        sorted_pairs = sorted(top100_pairs, key=lambda x: (-x[0], x[1], x[2]))
        seed_top100[str(seed)] = set((s, d) for _, s, d in sorted_pairs[:100])

        del all_scores_arr, top100_pairs
        gc.collect()

        print(f"    Seed {seed}: mean={score_stats['mean']:.4f}, "
              f"median={score_stats['median']:.4f}, "
              f"P95={score_stats['p95']:.4f}, P99={score_stats['p99']:.4f}, "
              f"score=1.0: {n_exact_1}, score>=0.9: {n_ge_09}")

    # Jaccard similarity between seed pairs
    jaccard_results = {}
    seed_list = list(seed_top100.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            s1, s2 = seed_list[i], seed_list[j]
            intersection = len(seed_top100[s1] & seed_top100[s2])
            union = len(seed_top100[s1] | seed_top100[s2])
            jaccard = intersection / max(union, 1)
            jaccard_results[f"{s1}_vs_{s2}"] = round(jaccard, 4)
            print(f"    Jaccard {s1} vs {s2}: {jaccard:.4f}")

    results["jaccard"] = jaccard_results
    del seed_top100
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# 3C: Score saturation audit
# ---------------------------------------------------------------------------
def score_saturation_audit(scores):
    """Count candidates at various score thresholds."""
    thresholds = [1.0, 0.99, 0.95, 0.90, 0.75, 0.50]
    results = {}
    total = len(scores)
    for t in thresholds:
        count = int((scores >= t).sum()) if t < 1.0 else int((scores == 1.0).sum())
        pct = count / max(total, 1) * 100
        results[f"score_{'=' if t == 1.0 else '>='}{t}"] = {"count": count, "percentage": round(pct, 2)}
        print(f"    Score {'= ' if t == 1.0 else '>= '}{t}: {count} ({pct:.2f}%)")
    return results


# ---------------------------------------------------------------------------
# 3D: Calibration analysis
# ---------------------------------------------------------------------------
def calibration_analysis(rf, X, id_to_idx, cs_test_pos, cs_test_neg, chunk_size=10000):
    """Evaluate RF calibration on cold-start held-out data."""
    print("  Calibration analysis...")
    from sklearn.metrics import brier_score_loss, log_loss

    n_pos = len(cs_test_pos)
    n_neg = len(cs_test_neg)

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
        xa = X[all_src[start:end]]
        xb = X[all_dst[start:end]]
        features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
        scores[start:end] = rf.predict_proba(features)[:, 1]
        del xa, xb, features

    # Brier score (lower is better, 0 is perfect)
    brier = float(brier_score_loss(labels, scores))

    # Log loss (clip scores to avoid log(0))
    scores_clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    logloss = float(log_loss(labels, scores_clipped))

    # Calibration curve (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(np.mean(scores[mask]))
            observed = float(np.mean(labels[mask]))
        else:
            mean_pred = (lo + hi) / 2
            observed = 0.0
        calibration.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": count,
            "mean_predicted": round(mean_pred, 4),
            "observed_positive_rate": round(observed, 4),
        })

    # ECE (Expected Calibration Error)
    ece = 0.0
    for bin_info in calibration:
        if bin_info["count"] > 0:
            w = bin_info["count"] / n_total
            ece += w * abs(bin_info["mean_predicted"] - bin_info["observed_positive_rate"])
    ece = round(float(ece), 4)

    results = {
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "ece": ece,
        "n_samples": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "calibration_curve": calibration,
    }

    print(f"    Brier score: {brier:.4f}")
    print(f"    Log loss: {logloss:.4f}")
    print(f"    ECE: {ece:.4f}")

    del all_src, all_dst, labels, scores, scores_clipped
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# 3F: Held-out positive rank distribution
# ---------------------------------------------------------------------------
def held_out_positive_rank_distribution(rf, X, id_to_idx, cs_test_pos,
                                        all_node_ids, sorted_edges,
                                        n_pos_sample=50000, n_neg_per_pos=50,
                                        seed=42, chunk_size=5000):
    """For each held-out positive, compute its rank among sampled candidates."""
    print("  Held-out positive rank distribution...")
    rng = np.random.default_rng(seed)

    # Subsample positives for speed
    n_total_pos = len(cs_test_pos)
    if n_pos_sample < n_total_pos:
        indices = rng.choice(n_total_pos, size=n_pos_sample, replace=False)
        sampled_pos = cs_test_pos[indices]
    else:
        sampled_pos = cs_test_pos
        n_pos_sample = n_total_pos

    ranks = []
    n_evaluated = 0

    # Build node set for quick lookup
    node_set = set(all_node_ids.tolist())

    for i in range(len(sampled_pos)):
        src_id = int(sampled_pos[i, 0])
        tgt_id = int(sampled_pos[i, 1])

        if src_id not in node_set or tgt_id not in node_set:
            continue

        # Sample negatives for this source
        neg_targets = []
        collected = 0
        attempts = 0
        while collected < n_neg_per_pos and attempts < n_neg_per_pos * 20:
            attempts += 1
            cand = int(rng.choice(all_node_ids))
            if cand == src_id or cand == tgt_id:
                continue
            if edge_exists(src_id, cand, sorted_edges):
                continue
            neg_targets.append(cand)
            collected += 1

        if len(neg_targets) == 0:
            continue

        # Score: positive + negatives
        all_targets = np.array([tgt_id] + neg_targets, dtype=np.int64)
        src_arr = np.full(len(all_targets), src_id, dtype=np.int64)
        src_idx = np.array([id_to_idx[int(s)] for s in src_arr], dtype=np.int64)
        dst_idx = np.array([id_to_idx[int(d)] for d in all_targets], dtype=np.int64)

        scores = score_pairs(rf, X, src_idx, dst_idx, chunk_size=chunk_size)

        # Rank: position of positive (index 0) in descending score order
        rank_order = np.argsort(-scores)
        pos_rank = int(np.where(rank_order == 0)[0][0]) + 1  # 1-indexed
        ranks.append(pos_rank)
        n_evaluated += 1

        if n_evaluated % 5000 == 0:
            print(f"    Processed {n_evaluated}/{n_pos_sample} positives...")

    ranks = np.array(ranks)
    results = {
        "n_evaluated": n_evaluated,
        "median_rank": float(np.median(ranks)) if len(ranks) > 0 else 0,
        "mean_rank": float(np.mean(ranks)) if len(ranks) > 0 else 0,
        "p10_rank": float(np.percentile(ranks, 10)) if len(ranks) > 0 else 0,
        "p25_rank": float(np.percentile(ranks, 25)) if len(ranks) > 0 else 0,
        "p50_rank": float(np.percentile(ranks, 50)) if len(ranks) > 0 else 0,
        "p75_rank": float(np.percentile(ranks, 75)) if len(ranks) > 0 else 0,
        "p90_rank": float(np.percentile(ranks, 90)) if len(ranks) > 0 else 0,
        "frac_ranked_1": float((ranks == 1).mean()) if len(ranks) > 0 else 0,
        "frac_in_top_10": float((ranks <= 10).mean()) if len(ranks) > 0 else 0,
        "frac_in_top_50": float((ranks <= 50).mean()) if len(ranks) > 0 else 0,
        "frac_in_top_100": float((ranks <= 100).mean()) if len(ranks) > 0 else 0,
    }

    print(f"    Evaluated {n_evaluated} held-out positives")
    print(f"    Median rank: {results['median_rank']:.1f}")
    print(f"    Fraction ranked #1: {results['frac_ranked_1']:.4f}")
    print(f"    Fraction in top-10: {results['frac_in_top_10']:.4f}")
    print(f"    Fraction in top-50: {results['frac_in_top_50']:.4f}")
    print(f"    Fraction in top-100: {results['frac_in_top_100']:.4f}")

    del ranks
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# 3G: Known edge exclusion audit
# ---------------------------------------------------------------------------
def exclusion_audit(candidates_csv_path, all_edges_sorted, id_to_idx, all_node_ids):
    """Verify no self-loops, known edges, duplicates, or invalid IDs."""
    print("  Known edge exclusion audit...")
    df = pd.read_csv(candidates_csv_path)

    src_ids = df["source_root_id"].values.astype(np.int64)
    dst_ids = df["target_root_id"].values.astype(np.int64)
    scores = df["predicted_score"].values.astype(np.float32)
    node_set = set(all_node_ids.tolist())

    # Self-loops
    self_loops = int((src_ids == dst_ids).sum())

    # Known edges
    known = ~verify_not_in_edges(src_ids, dst_ids, all_edges_sorted)
    known_edges = int(known.sum())

    # Duplicates
    pairs = set()
    n_dup = 0
    for s, d in zip(src_ids, dst_ids):
        key = (int(s), int(d))
        if key in pairs:
            n_dup += 1
        pairs.add(key)

    # Invalid IDs
    src_valid = np.array([s in node_set for s in src_ids])
    dst_valid = np.array([d in node_set for d in dst_ids])
    invalid_ids = int((~src_valid).sum() + (~dst_valid).sum())

    # Score checks
    finite = int(np.isfinite(scores).sum())
    not_finite = len(scores) - finite
    in_range = int(((scores >= 0) & (scores <= 1)).sum())
    out_of_range = len(scores) - in_range

    results = {
        "total_candidates": len(df),
        "self_loops": self_loops,
        "known_edges": known_edges,
        "duplicate_pairs": n_dup,
        "invalid_ids": invalid_ids,
        "non_finite_scores": not_finite,
        "out_of_range_scores": out_of_range,
    }

    print(f"    Total candidates: {len(df)}")
    print(f"    Self-loops: {self_loops}")
    print(f"    Known edges: {known_edges}")
    print(f"    Duplicate pairs: {n_dup}")
    print(f"    Invalid IDs: {invalid_ids}")
    print(f"    Non-finite scores: {not_finite}")
    print(f"    Out-of-range scores: {out_of_range}")

    ok = (self_loops == 0 and known_edges == 0 and n_dup == 0
          and invalid_ids == 0 and not_finite == 0 and out_of_range == 0)
    if ok:
        print("    ALL CHECKS PASSED")
    else:
        print("    WARNING: SOME CHECKS FAILED")

    del df, pairs
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# 3H: Post-hoc biological pattern analysis
# ---------------------------------------------------------------------------
def biological_pattern_analysis(candidates_csv_path, n_top=1000):
    """Analyze biological patterns in top candidates (post-hoc only)."""
    print("  Post-hoc biological pattern analysis...")
    nt = pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")
    nt_lookup = nt.set_index("root_id")[[
        "super_class", "primary_type", "nt_type", "flow",
    ]].to_dict("index")
    del nt
    gc.collect()

    df = pd.read_csv(candidates_csv_path).head(n_top)

    # Collect attributes
    src_nt, dst_nt = [], []
    src_sc, dst_sc = [], []
    src_flow, dst_flow = [], []

    for _, row in df.iterrows():
        si = nt_lookup.get(int(row["source_root_id"]), {})
        di = nt_lookup.get(int(row["target_root_id"]), {})
        src_nt.append(si.get("nt_type", "unknown") or "unknown")
        dst_nt.append(di.get("nt_type", "unknown") or "unknown")
        src_sc.append(si.get("super_class", "unknown") or "unknown")
        dst_sc.append(di.get("super_class", "unknown") or "unknown")
        src_flow.append(si.get("flow", "unknown") or "unknown")
        dst_flow.append(di.get("flow", "unknown") or "unknown")

    df["src_nt"] = src_nt
    df["dst_nt"] = dst_nt
    df["src_sc"] = src_sc
    df["dst_sc"] = dst_sc
    df["src_flow"] = src_flow
    df["dst_flow"] = dst_flow

    # NT type pair distribution
    nt_pairs = defaultdict(int)
    for s, d in zip(src_nt, dst_nt):
        nt_pairs[f"{s} -> {d}"] += 1

    # Super_class pair distribution
    sc_pairs = defaultdict(int)
    for s, d in zip(src_sc, dst_sc):
        sc_pairs[f"{s} -> {d}"] += 1

    # Flow pair distribution
    flow_pairs = defaultdict(int)
    for s, d in zip(src_flow, dst_flow):
        flow_pairs[f"{s} -> {d}"] += 1

    results = {
        "n_analyzed": len(df),
        "nt_type_pairs": dict(sorted(nt_pairs.items(), key=lambda x: -x[1])[:15]),
        "super_class_pairs": dict(sorted(sc_pairs.items(), key=lambda x: -x[1])[:15]),
        "flow_pairs": dict(sorted(flow_pairs.items(), key=lambda x: -x[1])[:10]),
    }

    print(f"    Analyzed top {len(df)} candidates")
    print(f"    Top NT type pairs:")
    for pair, count in sorted(nt_pairs.items(), key=lambda x: -x[1])[:5]:
        print(f"      {pair}: {count}")
    print(f"    Top super_class pairs:")
    for pair, count in sorted(sc_pairs.items(), key=lambda x: -x[1])[:5]:
        print(f"      {pair}: {count}")

    del df, nt_lookup
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Biological enrichment comparison (top-100 vs top-1000 vs full pool)
# ---------------------------------------------------------------------------
def biological_enrichment(candidates_csv_path, full_scores, full_src_ids, full_dst_ids,
                          id_to_idx, top_k_values=(100, 1000)):
    """Compare biological attribute distributions across score tiers."""
    print("  Biological enrichment analysis...")
    nt = pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")
    nt_lookup = nt.set_index("root_id")[["nt_type", "super_class", "flow"]].to_dict("index")
    del nt
    gc.collect()

    # Sort by score descending
    order = np.argsort(-full_scores)

    results = {}
    prev_mask = None
    for k in top_k_values:
        tier_order = order[:k]
        tier_src = full_src_ids[tier_order]
        tier_dst = full_dst_ids[tier_order]

        nt_dist = defaultdict(int)
        sc_dist = defaultdict(int)
        for s, d in zip(tier_src, tier_dst):
            si = nt_lookup.get(int(s), {})
            di = nt_lookup.get(int(d), {})
            src_nt = si.get("nt_type", "unknown") or "unknown"
            dst_nt = di.get("nt_type", "unknown") or "unknown"
            nt_dist[f"{src_nt} -> {dst_nt}"] += 1
            src_sc = si.get("super_class", "unknown") or "unknown"
            dst_sc = di.get("super_class", "unknown") or "unknown"
            sc_dist[f"{src_sc} -> {dst_sc}"] += 1

        results[f"top_{k}"] = {
            "nt_type_pairs": dict(sorted(nt_dist.items(), key=lambda x: -x[1])[:10]),
            "super_class_pairs": dict(sorted(sc_dist.items(), key=lambda x: -x[1])[:10]),
        }
        print(f"    Top-{k} NT pairs: {dict(sorted(nt_dist.items(), key=lambda x: -x[1])[:3])}")

    del nt_lookup
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment_3(smoke_test=False):
    print("=" * 70)
    print("EXPERIMENT 3: Candidate Ranking Robustness, Calibration, Plausibility")
    print("=" * 70)
    log_mem("program start")

    # ── Load data ──
    print("\n[0] Loading data...")
    data_dir = str(LP_DIR)
    cold_dir = str(LP_DIR / "cold_start")

    with open(MODELS_DIR / "link_prediction_rf.pkl", "rb") as f:
        rf_data = pickle.load(f)
    rf = rf_data["model"]
    feature_cols = rf_data["feature_cols"]
    log_mem("after loading RF model")

    X = np.load(os.path.join(data_dir, "X_features.npy"))
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    with open(os.path.join(data_dir, "id_to_idx.json")) as f:
        id_to_idx = {int(k): v for k, v in json.load(f).items()}
    all_node_ids = np.array(sorted(id_to_idx.keys()), dtype=np.int64)
    log_mem("after loading features")

    edges_df = pd.read_parquet(os.path.join(data_dir, "edges_aggregated.parquet"))
    all_edges = edges_df[["source", "target"]].values.astype(np.int64)
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    del edges_df
    gc.collect()
    log_mem("after loading edges")

    cs_test_pos = np.load(os.path.join(cold_dir, "split_test.npy"))
    cs_test_neg = np.load(os.path.join(cold_dir, "test_neg.npy"))
    log_mem("after loading cold-start test")

    print(f"  Nodes: {len(all_node_ids)}, Features: {X.shape[1]}")
    print(f"  Edges: {len(all_edges)}, RF estimators: {rf.n_estimators}")
    print(f"  Cold-start test: {len(cs_test_pos)} pos, {len(cs_test_neg)} neg")

    all_results = {}

    # ── 3A: Per-source ranking ──
    print("\n[3A] Per-source ranking evaluation...")
    n_src = 2000 if not smoke_test else 200
    per_source = per_source_ranking(
        rf, X, id_to_idx, cs_test_pos, cs_test_neg,
        all_node_ids, sorted_edges,
        n_sources=n_src, n_neg_per_source=100, seed=42,
    )
    all_results["3a_per_source_ranking"] = per_source
    log_mem("after 3A")

    # Random baseline
    print("\n[3A] Random baseline...")
    random_base = random_baseline_ranking(n_sources=n_src, n_neg_per_source=100)
    all_results["3a_random_baseline"] = random_base
    del sorted_edges
    gc.collect()
    log_mem("after random baseline")

    # ── 3B: Seed stability ──
    print("\n[3B] Candidate sampling robustness...")
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    n_src_s = 10000 if not smoke_test else 1000
    seed_results = seed_stability_analysis(
        rf, X, id_to_idx, all_node_ids, sorted_edges,
        n_sources=n_src_s, targets_per_source=100, seeds=(43, 44, 45),
    )
    all_results["3b_seed_stability"] = seed_results
    del sorted_edges
    gc.collect()
    log_mem("after 3B")

    # ── 3C: Score saturation (use Experiment 2D scores) ──
    print("\n[3C] Score saturation audit...")
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    all_cand_scores = []
    for scores_chunk in _generate_candidate_scores(
        rf, X, id_to_idx, all_node_ids, sorted_edges,
        n_sources=n_src_s, targets_per_source=100, seed=42,
    ):
        all_cand_scores.append(scores_chunk)
    del sorted_edges
    gc.collect()
    if all_cand_scores:
        saturation_scores = np.concatenate(all_cand_scores)
        saturation = score_saturation_audit(saturation_scores)
        all_results["3c_score_saturation"] = saturation
        del saturation_scores
    else:
        all_results["3c_score_saturation"] = {}
    del all_cand_scores
    gc.collect()
    log_mem("after 3C")

    # ── 3D: Calibration ──
    print("\n[3D] Calibration analysis...")
    calibration = calibration_analysis(rf, X, id_to_idx, cs_test_pos, cs_test_neg)
    all_results["3d_calibration"] = calibration
    log_mem("after 3D")

    # ── 3F: Held-out positive rank distribution ──
    print("\n[3F] Held-out positive rank distribution...")
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    n_pos_s = 50000 if not smoke_test else 5000
    rank_dist = held_out_positive_rank_distribution(
        rf, X, id_to_idx, cs_test_pos,
        all_node_ids, sorted_edges,
        n_pos_sample=n_pos_s, n_neg_per_pos=50, seed=42,
    )
    all_results["3f_rank_distribution"] = rank_dist
    del sorted_edges
    gc.collect()
    log_mem("after 3F")

    # ── 3G: Exclusion audit ──
    print("\n[3G] Known edge exclusion audit...")
    sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
    e2d_csv = Path.cwd() / "results" / "candidates" / "top_candidate_connections.csv"
    if e2d_csv.exists():
        audit = exclusion_audit(e2d_csv, sorted_edges, id_to_idx, all_node_ids)
        all_results["3g_exclusion_audit"] = audit
    else:
        all_results["3g_exclusion_audit"] = {"error": "candidates not found"}
    del sorted_edges
    gc.collect()
    log_mem("after 3G")

    # ── 3H: Post-hoc biological patterns ──
    print("\n[3H] Post-hoc biological pattern analysis...")
    if e2d_csv.exists():
        bio_patterns = biological_pattern_analysis(e2d_csv, n_top=1000)
        all_results["3h_biological_patterns"] = bio_patterns
    else:
        all_results["3h_biological_patterns"] = {}
    log_mem("after 3H")

    # ── Feature importance (reuse) ──
    print("\n[Feature importance] Reusing Experiment 2D results...")
    importances = rf.feature_importances_
    feat_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    n_feat = len(feat_pairs) // 4
    groups = {"source biology": 0.0, "target biology": 0.0,
              "source-target difference": 0.0, "source-target interaction": 0.0}
    for i, (_, imp) in enumerate(feat_pairs):
        if i < n_feat:
            groups["source biology"] += imp
        elif i < 2 * n_feat:
            groups["target biology"] += imp
        elif i < 3 * n_feat:
            groups["source-target difference"] += imp
        else:
            groups["source-target interaction"] += imp
    all_results["feature_importance"] = {
        "top10": [(n, float(v)) for n, v in feat_pairs[:10]],
        "groups": {k: float(v) for k, v in groups.items()},
    }
    for name, imp in feat_pairs[:10]:
        print(f"    {name:30s} {imp:.4f}")

    # ── 3E: Ranking vs probability interpretation ──
    print("\n[3E] Ranking vs probability interpretation...")
    brier = all_results.get("3d_calibration", {}).get("brier_score", 0)
    ece = all_results.get("3d_calibration", {}).get("ece", 0)
    roc = per_source.get("mean_recall@50", 0)
    ranking_interpretation = {
        "brier_score": brier,
        "ece": ece,
        "mean_recall_at_50": roc,
        "interpretation": (
            "The RF model is more appropriate as a ranking system than as a "
            "calibrated probability estimator." if brier > 0.05 or ece > 0.05
            else "The RF model shows reasonable calibration in addition to ranking ability."
        ),
    }
    all_results["3e_ranking_vs_probability"] = ranking_interpretation
    print(f"    {ranking_interpretation['interpretation']}")

    # ── Save results ──
    print("\n[Save] Writing results...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = REPORTS_DIR / "experiment_3_robustness.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    log_mem("final")
    print("\nDone!")
    return all_results


def _generate_candidate_scores(rf, X, id_to_idx, all_node_ids, sorted_edges,
                               n_sources, targets_per_source, seed):
    """Helper: generate and score candidates, yield (scores_array) chunks."""
    from src.link_prediction.experiment_2d import generate_candidates
    for src_chunk, dst_chunk in generate_candidates(
        all_node_ids, sorted_edges,
        n_sources=n_sources, targets_per_source=targets_per_source,
        seed=seed,
    ):
        all_scores = []
        for start in range(0, len(src_chunk), 5000):
            end = min(start + 5000, len(src_chunk))
            src_idx = np.array([id_to_idx[int(s)] for s in src_chunk[start:end]], dtype=np.int64)
            dst_idx = np.array([id_to_idx[int(d)] for d in dst_chunk[start:end]], dtype=np.int64)
            scores = score_pairs(rf, X, src_idx, dst_idx)
            all_scores.append(scores)
        yield np.concatenate(all_scores)
        del src_chunk, dst_chunk
        gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test (reduced scale)")
    args = parser.parse_args()
    run_experiment_3(smoke_test=args.smoke)
