# FlyMind Final Publication Freeze Report

**Date:** September 2026
**Branch:** docs/final-report-freeze
**Status:** READY WITH DOCUMENTED LIMITATIONS

---

## 1. Freeze Status

**READY WITH DOCUMENTED LIMITATIONS**

All scientific metrics have been verified against authoritative artifacts. Documentation inconsistencies have been corrected. The research results are now frozen.

---

## 2. Verified Metrics

### Cold-Start Evaluation (Primary Result)

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| RF Node Features ROC-AUC | 0.9800 | `results/reports/link_prediction_cold_start_results.json` → `rf_node_only.roc_auc` |
| RF Node Features PR-AUC | 0.9739 | `results/reports/link_prediction_cold_start_results.json` → `rf_node_only.pr_auc` |
| RF + Heuristics ROC-AUC | 0.6839 | `results/reports/link_prediction_cold_start_results.json` → `rf_node_plus_heuristic.roc_auc` |
| GraphSAGE ROC-AUC | 0.6485 | `results/reports/link_prediction_cold_start_results.json` → `gnn.roc_auc` |
| Random baseline ROC-AUC | 0.5007 | `results/reports/link_prediction_cold_start_results.json` → `random.roc_auc` |

### Random-Edge Evaluation

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| RF Node Features ROC-AUC | 0.9792 | `results/reports/link_prediction_results.json` → `rf_node_only.roc_auc` |
| RF + Heuristics ROC-AUC | 0.9905 | `results/reports/link_prediction_results.json` → `rf_node_plus_heuristic.roc_auc` |

### Candidate Ranking

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Recall@10 | 0.814 | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.mean_recall@10` |
| Hit Rate@10 | 0.995 | `results/reports/experiment_3_robustness.json` → `3a_per_source_ranking.hit_rate@10` |
| Validation ROC-AUC | 0.9796 | `results/reports/experiment_2d_results.json` → `validation.roc_auc` |

### Calibration

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Brier score | 0.0536 | `results/reports/experiment_3_robustness.json` → `3d_calibration.brier_score` |
| Log loss | 0.1914 | `results/reports/experiment_3_robustness.json` → `3d_calibration.log_loss` |
| ECE | 0.0411 | `results/reports/experiment_3_robustness.json` → `3d_calibration.ece` |

### Held-Out Ranking

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Median rank | 1.0 | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.median_rank` |
| Fraction ranked #1 | 53.6% | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.frac_ranked_1` |
| Fraction in top-10 | 95.9% | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution.frac_in_top_10` |

### Dataset Statistics

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Neurons | 139,255 | `data/processed/neuron_table.parquet` |
| Directed edges | 3,732,460 | `data/processed/link_prediction/edges_aggregated.parquet` |
| Unknown NT count | 19,658 | Calculated from `data/processed/neuron_table.parquet` |
| Unknown NT % | 14.1% | Calculated from `data/processed/neuron_table.parquet` |

---

## 3. Corrections Made

| # | Correction | Source of Error |
|---|------------|-----------------|
| 1 | Fixed Experiment 1 table with verified RF results from `baseline_results.json` | Incomplete table in original report |
| 2 | Fixed super-class list to match actual `super_class` column values | Incorrect class names |
| 3 | Fixed NT annotation counts: 14.1% unknown (19,658 neurons), not 97.3% | Confused candidate population with full dataset |
| 4 | Removed partial 0.982 ROC-AUC from Experiment 2C description | Aborted experiment metric should not appear |
| 5 | Fixed test count: 131 total (116 fast + 15 slow) | Inconsistent counting |
| 6 | Fixed candidate generation: 100 targets per source with 3x oversampling | Documentation conflict |
| 7 | Fixed chunk size: 50K for feature generation, 10K for candidate scoring | Documentation conflict |
| 8 | Updated software environment with verified package versions | Version verification |
| 9 | Updated all 97.3% NT references to clarify this refers to top candidates, not full dataset | Misleading statistics |

---

## 4. Remaining Limitations

These limitations are intentionally retained:

1. **Dataset-specific:** Results apply to FlyWire FAFB dataset only
2. **Incomplete NT annotations:** 14.1% of neurons lack neurotransmitter type labels
3. **No biological validation:** Candidate connections require experimental confirmation
4. **Ranking, not probability:** Scores are model-derived rankings, not calibrated probabilities
5. **Experiment 2C aborted:** Neural network ablation experiment was not completed
6. **Raw data external:** Source dataset lives outside the repository
7. **No pinned dependencies:** requirements.txt does not specify exact versions

---

## 5. Unverified Claims

| Claim | Status | Reason |
|-------|--------|--------|
| Majority baseline Accuracy/F1 | UNVERIFIED | Not stored in authoritative result files |
| Logistic Regression Accuracy/F1 | UNVERIFIED | Not stored in authoritative result files |
| GraphSAGE training time (306s) | UNVERIFIED | From `gnn_results.json` but not cross-validated |

All other major quantitative claims have been verified against authoritative artifacts.

---

## 6. Test Status

```
Collected: 131 tests
Passed: 131 tests (116 fast + 15 slow)
Failed: 0
Skipped: 0
```

Test suites:
- test_inference.py: 30 tests
- test_experiment_2d.py: 10 tests
- test_experiment_3.py: 16 tests
- test_data_integrity.py: 9 tests
- test_pipeline.py: 6 tests
- test_presentation.py: 26 tests
- test_dashboard.py: 19 tests
- test_link_prediction.py: 8 tests (slow, requires raw data)
- test_cold_start.py: 7 tests (slow, requires raw data)

---

## 7. Publication Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research report | `docs/FlyMind_Final_Research_Report.md` | ✅ Verified |
| Results traceability | `docs/RESULTS_TRACEABILITY.md` | ✅ Verified |
| Final results | `docs/FINAL_RESULTS.md` | ✅ Verified |
| Research audit | `docs/FINAL_RESEARCH_AUDIT.md` | ✅ Verified |
| Model artifact | `models/link_prediction_rf.pkl` | ✅ Present |
| Dashboard | `app/` | ✅ Functional |
| Presentation figures | `results/presentation/` | ✅ 6 figures |
| Test suite | `tests/` | ✅ 131 tests passing |

---

## 8. Scientific Freeze

> No further experimental changes are required for the current research version.

The following are now frozen:
- All model architectures and hyperparameters
- All evaluation protocols and splits
- All reported metrics and results
- Feature definitions and engineering
- Negative sampling methodology

---

## 9. Allowed Changes After Freeze

Only the following changes are permitted:
- Typo corrections
- Formatting improvements
- Documentation clarifications
- Presentation/UI improvements
- Packaging and deployment
- README updates
- Citation corrections

Scientific metrics and evaluation methodology are frozen and must not be changed.
