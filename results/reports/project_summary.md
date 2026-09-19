# FlyMind Link Prediction: Project Summary

## Dataset

- 139,255 neurons, 3,732,460 directed edges
- 15 safe features: 13 numeric + 2 categorical
- Pair features: [x_A, x_B, |x_A-x_B|, x_A*x_B] = 60 dimensions
- Cold-start test: 1,043,974 held-out positive edges

## Experiment 1: Super-Class Classification (GNN)

- GraphSAGE model predicting super_class from graph structure
- Best model saved at `models/graphsage_superclass_best.pt`
- Serves as foundation for node embeddings

## Experiment 2 Part 1: Random Edge Holdout

| Model | ROC-AUC |
|-------|--------:|
| RF Node Features | 0.979 |
| RF + Heuristic | 0.990 |

## Experiment 2B: Cold-Start Prediction

| Model | ROC-AUC | PR-AUC |
|-------|--------:|-------:|
| RF Node Features | 0.980 | 0.974 |
| RF + Heuristic | 0.684 | - |
| GraphSAGE | 0.649 | - |

Sanity test AUC: 0.999

## Experiment 2D: Candidate Connection Ranking

Used validated RF model to rank 1M candidate directed connections (10K sources x 100 targets).

| Metric | Value |
|--------|------:|
| ROC-AUC (validation) | 0.9796 |
| PR-AUC (validation) | 0.9740 |
| Candidates scored | 1,000,000 |
| Final top candidates | 100,000 |
| Peak RAM | 1,056 MB |

Score distribution: mean 0.1346, median 0.03, P95 0.68, P99 0.91

Top features: area_nm (0.020), length_nm (0.017), size_nm (0.015), coord_x (0.014)

Feature groups: source biology (0.052) > target biology (0.033) > interaction (0.025) > difference (0.023)

## Experiment 3: Robustness, Calibration, Biological Plausibility

### 3A: Per-Source Ranking

| Model | Recall@10 | Recall@50 | Hit Rate@10 | Hit Rate@50 |
|-------|----------:|----------:|------------:|------------:|
| RF Node Features | 0.814 | 0.990 | 0.995 | 1.000 |
| Random Baseline | 0.098 | 0.463 | 0.420 | 0.960 |

RF is 8.3x better than random at K=10.

### 3B: Seed Stability

Score distributions stable across seeds 42-45 (mean varies by <0.003). Jaccard = 0.0 for top-100 pairs (expected: different seeds sample different sources).

### 3C: Score Saturation

| Threshold | Count | Pct |
|-----------|------:|----:|
| score = 1.0 | 34 | 0.03% |
| score >= 0.90 | 1,103 | 1.10% |
| score >= 0.50 | 9,257 | 9.26% |

Saturation is minimal. Model is genuinely discriminative.

### 3D: Calibration

| Metric | Value |
|--------|------:|
| Brier score | 0.0536 |
| Log loss | 0.1914 |
| ECE | 0.0411 |

Reasonably calibrated. Slightly overconfident at low probabilities, slightly underconfident at high probabilities.

### 3E: Ranking vs Probability

> The RF model is more appropriate as a ranking system than as a calibrated probability estimator.

### 3F: Held-out Positive Rank Distribution

| Metric | Value |
|--------|------:|
| Median rank | 1.0 |
| Mean rank | 2.8 |
| Fraction ranked #1 | 53.6% |
| Fraction in top-10 | 95.9% |
| Fraction in top-50 | 100.0% |

### 3G: Exclusion Audit

All checks passed: 0 self-loops, 0 known edges, 0 duplicates, 0 invalid IDs, 0 non-finite scores.

### 3H: Post-Hoc Biological Patterns

97.3% of top candidates have unknown NT types (19K neurons lack NT annotation). Known-type pairs show enrichment for GABA and ACH.

## Test Suite

41/41 tests passing across 4 test files.

| Suite | Tests | Status |
|-------|------:|--------|
| test_link_prediction.py | 8 | PASS |
| test_cold_start.py | 7 | PASS |
| test_experiment_2d.py | 10 | PASS |
| test_experiment_3.py | 16 | PASS |

## Key Files

| File | Purpose |
|------|---------|
| `src/link_prediction/config.py` | Paths, features, constants |
| `src/link_prediction/models.py` | RF training, GraphSAGE encoder |
| `src/link_prediction/features.py` | Chunked pair features |
| `src/link_prediction/evaluate.py` | ROC-AUC, PR-AUC, P@K |
| `src/link_prediction/negative_sampling.py` | Compact hash-based edge index |
| `src/link_prediction/experiment_2d.py` | Candidate ranking pipeline |
| `src/link_prediction/experiment_3.py` | Robustness, calibration, analysis |
| `models/link_prediction_rf.pkl` | Trained RF model (185 MB) |
| `results/candidates/top_candidate_connections.csv` | Top 100K candidates |
| `results/candidates/top20_annotated.csv` | Top 20 with annotations |

## Conclusions

1. The RF model discriminates held-out connections from non-edges (ROC-AUC 0.980).
2. Per-source ranking is substantially better than random (81.4% vs 9.8% recall at K=10).
3. 53.6% of true held-out edges rank #1 among sampled candidates.
4. Score distributions are stable across random seeds.
5. The model is reasonably calibrated (ECE 0.041) but best used as a ranker.
6. Score saturation is minimal (0.03% at 1.0).
7. Candidate set shows enrichment for intrinsic flow and spatial proximity.
8. Limitations: only node features, no structural features, NT type unknown for 14% of neurons.

## Scientific Language

All candidate connections are described as:

> model-suggested candidate connections

Never as "discovered" or "predicted" biological connections.

## Memory

All experiments run under 1.1 GB peak RSS on a 5.5 GB machine. No all-pairs enumeration. Seeds processed sequentially with gc.collect().
