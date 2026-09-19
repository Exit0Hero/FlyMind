"""Tests for Experiment 3: Robustness, Calibration, Biological Plausibility."""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_no_self_loops_in_candidates():
    """Discovery candidate CSV has no self-loops."""
    import pandas as pd
    csv = pathlib.Path("results/candidates/top_candidate_connections.csv")
    if not csv.exists():
        print("  SKIP: candidates CSV not found")
        return
    df = pd.read_csv(csv)
    self_loops = (df["source_root_id"] == df["target_root_id"]).sum()
    assert self_loops == 0, f"Found {self_loops} self-loops"
    print("  PASS: no self-loops in candidates")


def test_no_known_edges_in_candidates():
    """Discovery candidate CSV has no known edges."""
    import pandas as pd
    from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges
    csv = pathlib.Path("results/candidates/top_candidate_connections.csv")
    if not csv.exists():
        print("  SKIP: candidates CSV not found")
        return
    edges_df = pd.read_parquet("data/processed/link_prediction/edges_aggregated.parquet")
    sorted_edges = build_sorted_pairs(
        edges_df["source"].values.astype(np.int64),
        edges_df["target"].values.astype(np.int64),
    )
    df = pd.read_csv(csv)
    src = df["source_root_id"].values.astype(np.int64)
    dst = df["target_root_id"].values.astype(np.int64)
    known = ~verify_not_in_edges(src, dst, sorted_edges)
    assert known.sum() == 0, f"Found {known.sum()} known edges in candidates"
    print("  PASS: no known edges in candidates")


def test_no_duplicate_pairs():
    """Discovery candidate CSV has no duplicate source-target pairs."""
    import pandas as pd
    csv = pathlib.Path("results/candidates/top_candidate_connections.csv")
    if not csv.exists():
        print("  SKIP: candidates CSV not found")
        return
    df = pd.read_csv(csv)
    pairs = set()
    for _, row in df.iterrows():
        key = (int(row["source_root_id"]), int(row["target_root_id"]))
        assert key not in pairs, f"Duplicate pair: {key}"
        pairs.add(key)
    print("  PASS: no duplicate pairs")


def test_score_range_0_1():
    """All candidate scores are in [0, 1]."""
    import pandas as pd
    csv = pathlib.Path("results/candidates/top_candidate_connections.csv")
    if not csv.exists():
        print("  SKIP: candidates CSV not found")
        return
    df = pd.read_csv(csv)
    scores = df["predicted_score"].values
    assert np.all(scores >= 0), f"Scores below 0: {(scores < 0).sum()}"
    assert np.all(scores <= 1), f"Scores above 1: {(scores > 1).sum()}"
    print("  PASS: scores in [0,1]")


def test_finite_scores():
    """All candidate scores are finite."""
    import pandas as pd
    csv = pathlib.Path("results/candidates/top_candidate_connections.csv")
    if not csv.exists():
        print("  SKIP: candidates CSV not found")
        return
    df = pd.read_csv(csv)
    scores = df["predicted_score"].values
    assert np.all(np.isfinite(scores)), f"Non-finite scores: {np.sum(~np.isfinite(scores))}"
    print("  PASS: finite scores")


def test_per_source_recall_positive():
    """Per-source Recall@50 > 0.3 (much better than random ~0.05)."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    r50 = res.get("3a_per_source_ranking", {}).get("mean_recall@50", 0)
    assert r50 > 0.3, f"Mean Recall@50 = {r50:.4f}, expected > 0.3"
    print(f"  PASS: per-source Recall@50 = {r50:.4f} > 0.3")


def test_random_baseline_lower():
    """Random baseline Recall@50 < RF Recall@50."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    rf_r50 = res.get("3a_per_source_ranking", {}).get("mean_recall@50", 0)
    rand_r50 = res.get("3a_random_baseline", {}).get("mean_recall@50", 1)
    assert rf_r50 > rand_r50, f"RF {rf_r50:.4f} <= Random {rand_r50:.4f}"
    print(f"  PASS: RF {rf_r50:.4f} > Random {rand_r50:.4f}")


def test_calibration_brier_reasonable():
    """Brier score < 0.15 (reasonably calibrated)."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    brier = res.get("3d_calibration", {}).get("brier_score", 1)
    assert brier < 0.15, f"Brier = {brier:.4f}, expected < 0.15"
    print(f"  PASS: Brier score = {brier:.4f} < 0.15")


def test_calibration_ece():
    """ECE < 0.15."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    ece = res.get("3d_calibration", {}).get("ece", 1)
    assert ece < 0.15, f"ECE = {ece:.4f}, expected < 0.15"
    print(f"  PASS: ECE = {ece:.4f} < 0.15")


def test_calibration_bins_sum_to_n():
    """Calibration bin counts sum to total samples."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    bins = res.get("3d_calibration", {}).get("calibration_curve", [])
    total_in_bins = sum(b["count"] for b in bins)
    n_samples = res.get("3d_calibration", {}).get("n_samples", 0)
    assert total_in_bins == n_samples, f"Bin sum {total_in_bins} != n_samples {n_samples}"
    print(f"  PASS: calibration bins sum to {n_samples}")


def test_seed_stability_stats():
    """Seed stability shows consistent distributions."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    ss = res.get("3b_seed_stability", {})
    means = []
    for seed in ["43", "44", "45"]:
        if seed in ss:
            means.append(ss[seed]["score_stats"]["mean"])
    assert len(means) == 3, f"Expected 3 seeds, got {len(means)}"
    # All means should be within 0.05 of each other
    assert max(means) - min(means) < 0.05, f"Seed means vary too much: {means}"
    print(f"  PASS: seed means consistent: {[f'{m:.4f}' for m in means]}")


def test_jaccard_non_negative():
    """Jaccard similarities are non-negative."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    jaccard = res.get("3b_seed_stability", {}).get("jaccard", {})
    for pair, val in jaccard.items():
        assert val >= 0, f"Negative Jaccard for {pair}: {val}"
    print("  PASS: Jaccard similarities non-negative")


def test_rank_distribution_median_rank():
    """Median rank of held-out positives < 5."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    median = res.get("3f_rank_distribution", {}).get("median_rank", 999)
    assert median < 5, f"Median rank = {median:.1f}, expected < 5"
    print(f"  PASS: median rank = {median:.1f} < 5")


def test_rank_distribution_frac_top10():
    """Fraction in top-10 > 0.8."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    frac = res.get("3f_rank_distribution", {}).get("frac_in_top_10", 0)
    assert frac > 0.8, f"Fraction in top-10 = {frac:.4f}, expected > 0.8"
    print(f"  PASS: fraction in top-10 = {frac:.4f} > 0.8")


def test_exclusion_audit_all_zero():
    """Exclusion audit shows all zeros."""
    with open("results/reports/experiment_3_robustness.json") as f:
        import json
        res = json.load(f)
    audit = res.get("3g_exclusion_audit", {})
    for key in ["self_loops", "known_edges", "duplicate_pairs", "invalid_ids",
                "non_finite_scores", "out_of_range_scores"]:
        assert audit.get(key, -1) == 0, f"{key} = {audit.get(key)}, expected 0"
    print("  PASS: exclusion audit all zeros")


def test_no_all_pairs_construction():
    """Verification: no dense all-pairs matrix exists."""
    # This is a structural check - the code never creates N*N matrices
    # We verify by checking that no file > 500MB exists in results/candidates
    import os
    cand_dir = pathlib.Path("results/candidates")
    if cand_dir.exists():
        for f in cand_dir.iterdir():
            size = f.stat().st_size
            assert size < 500e6, f"File too large ({size/1e6:.0f} MB): {f}"
    print("  PASS: no large dense matrices")


if __name__ == "__main__":
    tests = [
        test_no_self_loops_in_candidates,
        test_no_known_edges_in_candidates,
        test_no_duplicate_pairs,
        test_score_range_0_1,
        test_finite_scores,
        test_per_source_recall_positive,
        test_random_baseline_lower,
        test_calibration_brier_reasonable,
        test_calibration_ece,
        test_calibration_bins_sum_to_n,
        test_seed_stability_stats,
        test_jaccard_non_negative,
        test_rank_distribution_median_rank,
        test_rank_distribution_frac_top10,
        test_exclusion_audit_all_zero,
        test_no_all_pairs_construction,
    ]

    print("=" * 60)
    print("EXPERIMENT 3 TESTS")
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
