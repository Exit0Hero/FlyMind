# FlyMind Results Traceability

This document maps every major quantitative claim in the research report to its authoritative source artifact.

---

## Dataset Statistics

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Neuron count | 139,255 | Processed neuron table | `data/processed/neuron_table.parquet` |
| Directed edge count | 3,732,460 | Aggregated edges | `data/processed/link_prediction/edges_aggregated.parquet` |
| Raw edge rows | 5,342,446 | Princeton connections | `data/processed/edges_princeton.parquet` |
| Brain regions | 76 | Neuron table metadata | `data/processed/neuron_table.parquet` |
| NT types | 6 | Neuron table `nt_type` | `data/processed/neuron_table.parquet` |
| Super-classes | 10 | Neuron table `super_class` | `data/processed/neuron_table.parquet` |
| Unknown NT % | 97.3% | Calculated from neuron table | `data/processed/neuron_table.parquet` |
| Missing morphology | 27 neurons | Feature matrix NaN count | `data/processed/link_prediction/X_features.npy` |

---

## Cold-Start Split

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Train nodes | 97,478 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.train_nodes` |
| Val nodes | 20,888 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.val_nodes` |
| Test nodes | 20,889 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.test_nodes` |
| Train edges | 1,832,606 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.train_edges` |
| Val edges | 855,880 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.val_edges` |
| Test edges | 1,043,974 | Cold-start split | `results/reports/link_prediction_cold_start_results.json` → `split_sizes.test_edges` |

---

## Experiment 2A — Random Edge Holdout

| Claim | Value | Source | File |
|-------|-------|--------|------|
| RF Node Features ROC-AUC | 0.979 | Random edge results | `results/reports/link_prediction_results.json` → `rf_node_only.roc_auc` |
| RF Node Features PR-AUC | 0.973 | Random edge results | `results/reports/link_prediction_results.json` → `rf_node_only.pr_auc` |
| RF + Heuristics ROC-AUC | 0.990 | Random edge results | `results/reports/link_prediction_results.json` → `rf_node_plus_heuristic.roc_auc` |
| RF + Heuristics PR-AUC | 0.989 | Random edge results | `results/reports/link_prediction_results.json` → `rf_node_plus_heuristic.pr_auc` |
| Random baseline ROC-AUC | 0.500 | Random edge results | `results/reports/link_prediction_results.json` → `random.roc_auc` |

---

## Cold-Start Evaluation

| Claim | Value | Source | File |
|-------|-------|--------|------|
| RF Node Features ROC-AUC | 0.980 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `rf_node_only.roc_auc` |
| RF Node Features PR-AUC | 0.974 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `rf_node_only.pr_auc` |
| RF + Heuristics ROC-AUC | 0.684 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `rf_node_plus_heuristic.roc_auc` |
| RF + Heuristics PR-AUC | 0.576 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `rf_node_plus_heuristic.pr_auc` |
| GraphSAGE ROC-AUC | 0.649 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `gnn.roc_auc` |
| GraphSAGE PR-AUC | 0.646 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `gnn.pr_auc` |
| Random baseline ROC-AUC | 0.501 | Cold-start results | `results/reports/link_prediction_cold_start_results.json` → `random.roc_auc` |

---

## Experiment 2D — Candidate Ranking

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Validation ROC-AUC | 0.9796 | Candidate ranking results | `results/reports/experiment_2d_results.json` → `validation.roc_auc` |
| Validation PR-AUC | 0.9740 | Candidate ranking results | `results/reports/experiment_2d_results.json` → `validation.pr_auc` |
| Sources sampled | 10,000 | Candidate generation | `results/reports/experiment_2d_results.json` → `candidate_generation.n_sources` |
| Candidates scored | 1,000,000 | Candidate generation | `results/reports/experiment_2d_results.json` → `candidate_generation.total_candidates_scored` |
| Final top candidates | 100,000 | Candidate generation | `results/reports/experiment_2d_results.json` → `candidate_generation.final_top_candidates` |

---

## Experiment 3 — Robustness

### Per-Source Ranking (3A)

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Recall@10 (RF) | 0.814 | Per-source ranking | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.mean_recall@10` |
| Recall@50 (RF) | 0.990 | Per-source ranking | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.mean_recall@50` |
| Hit Rate@10 (RF) | 0.995 | Per-source ranking | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.hit_rate@10` |
| Hit Rate@50 (RF) | 1.000 | Per-source ranking | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.hit_rate@50` |
| Recall@10 (Random) | 0.098 | Random baseline | `results/reports/experiment_3_robustness.json` → `3a_random_baseline.mean_recall@10` |
| Recall@50 (Random) | 0.463 | Random baseline | `results/reports/experiment_3_robustness.json` → `3a_random_baseline.mean_recall@50` |
| Hit Rate@10 (Random) | 0.420 | Random baseline | `results/reports/experiment_3_robustness.json` → `3a_random_baseline.hit_rate@10` |
| Hit Rate@50 (Random) | 0.960 | Random baseline | `results/reports/experiment_3_robustness.json` → `3a_random_baseline.hit_rate@50` |

### Calibration (3D)

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Brier score | 0.0536 | Calibration | `results/reports/experiment_3_robustness.json` → `3d_calibration.brier_score` |
| Log loss | 0.1914 | Calibration | `results/reports/experiment_3_robustness.json` → `3d_calibration.log_loss` |
| ECE | 0.0411 | Calibration | `results/reports/experiment_3_robustness.json` → `3d_calibration.ece` |

### Score Saturation (3C)

| Claim | Value | Source | File |
|-------|-------|--------|------|
| score = 1.0 | 34 (0.03%) | Score saturation | `results/reports/experiment_3_robustness.json` → `3c_score_saturation.score_=1.0` |
| score ≥ 0.90 | 1,103 (1.10%) | Score saturation | `results/reports/experiment_3_robustness.json` → `3c_score_saturation.score_>=0.9` |
| score ≥ 0.50 | 9,257 (9.26%) | Score saturation | `results/reports/experiment_3_robustness.json` → `3c_score_saturation.score_>=0.5` |

### Held-Out Positive Ranking (3F)

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Median rank | 1.0 | Rank distribution | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.median_rank` |
| Mean rank | 2.8 | Rank distribution | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.mean_rank` |
| Fraction ranked #1 | 53.6% | Rank distribution | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.frac_ranked_1` |
| Fraction in top-10 | 95.9% | Rank distribution | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.frac_in_top_10` |
| Fraction in top-50 | 100.0% | Rank distribution | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.frac_in_top_50` |

### Exclusion Audit (3G)

| Claim | Value | Source | File |
|-------|-------|--------|------|
| Self-loops | 0 | Exclusion audit | `results/reports/experiment_3_robustness.json` → `3g_exclusion_audit.self_loops` |
| Known edges | 0 | Exclusion audit | `results/reports/experiment_3_robustness.json` → `3g_exclusion_audit.known_edges` |
| Duplicates | 0 | Exclusion audit | `results/reports/experiment_3_robustness.json` → `3g_exclusion_audit.duplicate_pairs` |
| Invalid IDs | 0 | Exclusion audit | `results/reports/experiment_3_robustness.json` → `3g_exclusion_audit.invalid_ids` |
| Non-finite scores | 0 | Exclusion audit | `results/reports/experiment_3_robustness.json` → `3g_exclusion_audit.non_finite_scores` |

---

## Feature Importance

| Claim | Value | Source | File |
|-------|-------|--------|------|
| area_nm importance | 0.0203 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| length_nm importance | 0.0165 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| size_nm importance | 0.0152 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| coord_x importance | 0.0142 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| coord_y importance | 0.0100 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| gaba_avg importance | 0.0088 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| coord_z importance | 0.0085 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| ach_avg importance | 0.0073 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| da_avg importance | 0.0070 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |
| nt_type_score importance | 0.0059 | Feature importance | `results/reports/experiment_2d_results.json` → `feature_importance_top10` |

---

## Experiment 1 — Super-Class Classification

| Claim | Value | Source | File |
|-------|-------|--------|------|
| GraphSAGE Accuracy | 0.9265 | Classification results | `results/reports/link_prediction_results.json` (if present) |
| GraphSAGE Weighted F1 | 0.9372 | Classification results | `results/reports/link_prediction_results.json` (if present) |

---

## Model Metadata

| Claim | Value | Source | File |
|-------|-------|--------|------|
| RF estimators | 100 | Model config | `src/link_prediction/config.py` |
| Random seed | 42 | Config | `src/link_prediction/config.py` |
| Pair feature dim | 60 | Feature construction | `src/link_prediction/features.py` |
| Node features | 15 | Feature list | `src/link_prediction/config.py` |
| Model file size | 185 MB | File system | `models/link_prediction_rf.pkl` |

---

## Test Results

| Claim | Value | Source | File |
|-------|-------|--------|------|
| test_inference.py | 30 tests | Test suite | `tests/test_inference.py` |
| test_experiment_2d.py | 10 tests | Test suite | `tests/test_experiment_2d.py` |
| test_experiment_3.py | 16 tests | Test suite | `tests/test_experiment_3.py` |
| test_data_integrity.py | 9 tests | Test suite | `tests/test_data_integrity.py` |
| test_pipeline.py | 6 tests | Test suite | `tests/test_pipeline.py` |
| test_presentation.py | 26 tests | Test suite | `tests/test_presentation.py` |
| test_dashboard.py | 19 tests | Test suite | `tests/test_dashboard.py` |
| **Total** | **116 tests** | All suites | `tests/` |

---

## Consistency Notes

1. **Cold-start ROC-AUC** appears as 0.9800 in `FINAL_RESULTS.md` and 0.980 in the research report. The authoritative value is 0.980017... from `link_prediction_cold_start_results.json`.

2. **Random-edge RF + Heuristics** ROC-AUC appears as 0.9905 in `FINAL_RESULTS.md` and 0.990 in the research report. The authoritative value is 0.9905... from `link_prediction_results.json`.

3. **Experiment 2C** is explicitly documented as aborted and is NOT included in any results summary.

4. **All candidate connections** are described as "model-suggested candidate connections" throughout all documentation.

5. **Ranking scores** are never described as literal biological probabilities.
