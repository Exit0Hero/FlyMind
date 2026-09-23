# Phase 7 Final Report — FlyMind Publication Freeze & Verification

**Date:** September 2026
**Branch:** `docs/final-report-freeze`
**Commit:** `c1911d8`
**Status:** READY WITH DOCUMENTED LIMITATIONS

---

## 1. Executive Summary

Phase 7 completed a comprehensive verification and consistency audit of all FlyMind research documentation against authoritative artifacts. Nine categories of inconsistencies were identified and corrected. The research results are now frozen and publication-ready.

---

## 2. Phase 7 Status

**READY WITH DOCUMENTED LIMITATIONS**

All scientific metrics verified. Documentation inconsistencies corrected. Scientific freeze imposed.

---

## 3. Documents Updated

| # | Document | Changes Made |
|---|----------|--------------|
| 1 | `docs/FlyMind_Final_Research_Report.md` | Experiment 1 table, super-class list, NT counts, Experiment 2C language, test count, candidate params, chunk size, environment |
| 2 | `docs/RESULTS_TRACEABILITY.md` | NT counts, test counts for all suites |
| 3 | `docs/FINAL_PUBLICATION_FREEZE.md` | **NEW** — Publication freeze report |
| 4 | `docs/FINAL_RESEARCH_AUDIT.md` | NT counts |
| 5 | `docs/FINAL_RESULTS.md` | NT counts |
| 6 | `docs/INFERENCE_ARCHITECTURE.md` | NT counts |
| 7 | `docs/PRESENTATION_DATA_LAYER.md` | NT counts |
| 8 | `docs/DEMO_SCRIPT.md` | NT counts |
| 9 | `docs/JUDGE_QA.md` | NT counts |
| 10 | `README.md` | NT counts |
| 11 | `app/pages/about.py` | NT counts |
| 12 | `app/pages/neuron_explorer.py` | NT counts |

---

## 4. Corrections Made

### 4.1 Experiment 1 Table

**Before:** Incomplete table with only GraphSAGE results populated.

**After:** Verified RF results from `results/reports/baseline_results.json`.

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| RF Biology | 0.9727 | 0.9733 | 0.8724 |
| RF Graph-only | 0.6655 | 0.6816 | 0.3996 |
| RF Bio + Graph | 0.9773 | 0.9775 | 0.9005 |
| GraphSAGE | 0.9265 | 0.9372 | 0.7066 |

**Source:** `results/reports/baseline_results.json`, `results/reports/gnn_results.json`

### 4.2 Super-Class List

**Before:** motor, sensory, central, visual_projection, visual_centrifugal, optic, ascending, descending, unknown, other

**After:** optic, central, sensory, visual_projection, ascending, descending, sensory_ascending, visual_centrifugal, motor, endocrine

**Source:** `data/processed/neuron_table.parquet` → `super_class.value_counts()`

### 4.3 Neurotransmitter Counts

**Before:** "97.3% of neurons (approximately 135,500 of 139,255) have unknown neurotransmitter annotations"

**After:** "14.1% of neurons (19,658) lack neurotransmitter type annotations"

**Source:** `data/processed/neuron_table.parquet`

| NT Type | Count | Percentage |
|---------|------:|-----------:|
| ACH | 82,298 | 59.1% |
| GLUT | 19,605 | 14.1% |
| GABA | 16,017 | 11.5% |
| SER | 1,021 | 0.7% |
| DA | 584 | 0.4% |
| OCT | 72 | 0.1% |
| **Unknown** | **19,658** | **14.1%** |

**Note:** The 97.3% figure refers to the proportion of *top candidates* involving neurons with unknown NT types, not the full dataset. This was clarified in all references.

### 4.4 Experiment 2C Language

**Before:** "The MLP cold-start evaluation completed partially (ROC-AUC 0.982) but GraphSAGE crashed before completion."

**After:** "The experiment did not complete successfully."

**Rationale:** Partial metrics from aborted experiments should not appear in publication-facing documents.

### 4.5 Test Count

**Before:** "Total: 101 PASS"

**After:** "Total: 131 tests (116 fast + 15 slow) — All PASS"

| Suite | Tests | Notes |
|-------|------:|-------|
| test_inference.py | 30 | Fast |
| test_experiment_2d.py | 10 | Fast |
| test_experiment_3.py | 16 | Fast |
| test_data_integrity.py | 9 | Fast |
| test_pipeline.py | 6 | Fast |
| test_presentation.py | 26 | Fast |
| test_dashboard.py | 19 | Fast |
| test_link_prediction.py | 8 | Slow (raw data) |
| test_cold_start.py | 7 | Slow (raw data) |
| **Total** | **131** | |

### 4.6 Candidate Generation Parameters

**Before:** "Sample candidate pool (1000 random targets per source)"

**After:** "Sample candidate pool (100 targets per source, with 3x oversampling for filtering)"

**Source:** `src/link_prediction/experiment_2d.py` → `targets_per_source=100`

### 4.7 Chunk Size

**Before:** "Chunked pair feature construction (10K pairs/chunk)"

**After:** "Chunked pair feature construction (50K pairs/chunk for feature generation, 10K for candidate scoring)"

**Source:** `src/link_prediction/features.py` → `chunk_size=50000`, `src/link_prediction/experiment_2d.py` → `chunk_size=10000`

### 4.8 Software Environment

| Package | Version | Verified |
|---------|---------|----------|
| Python | 3.14.4 | ✅ |
| PyTorch | 2.14.0+cpu | ✅ |
| PyG | 2.8.0.post1 | ✅ |
| Streamlit | 1.64.0 | ✅ |
| Scikit-learn | 1.9.1 | ✅ |
| NumPy | 2.5.3 | ✅ |

---

## 5. Verified Metrics

### Cold-Start Evaluation (Primary Result)

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| RF Node Features ROC-AUC | 0.9800 | `link_prediction_cold_start_results.json` → `rf_node_only.roc_auc` |
| RF Node Features PR-AUC | 0.9739 | `link_prediction_cold_start_results.json` → `rf_node_only.pr_auc` |
| RF + Heuristics ROC-AUC | 0.6839 | `link_prediction_cold_start_results.json` → `rf_node_plus_heuristic.roc_auc` |
| GraphSAGE ROC-AUC | 0.6485 | `link_prediction_cold_start_results.json` → `gnn.roc_auc` |
| Random baseline ROC-AUC | 0.5007 | `link_prediction_cold_start_results.json` → `random.roc_auc` |

### Random-Edge Evaluation

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| RF Node Features ROC-AUC | 0.9792 | `link_prediction_results.json` → `rf_node_only.roc_auc` |
| RF + Heuristics ROC-AUC | 0.9905 | `link_prediction_results.json` → `rf_node_plus_heuristic.roc_auc` |

### Candidate Ranking

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Recall@10 | 0.814 | `experiment_3_robustness.json` → `3a_per_source_ranking.mean_recall@10` |
| Hit Rate@10 | 0.995 | `experiment_3_robustness.json` → `3a_per_source_ranking.hit_rate@10` |
| Validation ROC-AUC | 0.9796 | `experiment_2d_results.json` → `validation.roc_auc` |
| Validation PR-AUC | 0.9740 | `experiment_2d_results.json` → `validation.pr_auc` |

### Calibration

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Brier score | 0.0536 | `experiment_3_robustness.json` → `3d_calibration.brier_score` |
| Log loss | 0.1914 | `experiment_3_robustness.json` → `3d_calibration.log_loss` |
| ECE | 0.0411 | `experiment_3_robustness.json` → `3d_calibration.ece` |

### Held-Out Ranking

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Median rank | 1.0 | `experiment_3_robustness.json` → `3f_rank_distribution.median_rank` |
| Mean rank | 2.8 | `experiment_3_robustness.json` → `3f_rank_distribution.mean_rank` |
| Fraction ranked #1 | 53.6% | `experiment_3_robustness.json` → `3f_rank_distribution.frac_ranked_1` |
| Fraction in top-10 | 95.9% | `experiment_3_robustness.json` → `3f_rank_distribution.frac_in_top_10` |
| Fraction in top-50 | 100.0% | `experiment_3_robustness.json` → `3f_rank_distribution.frac_in_top_50` |

### Dataset Statistics

| Metric | Value | Source Artifact |
|--------|-------|-----------------|
| Neurons | 139,255 | `data/processed/neuron_table.parquet` |
| Directed edges | 3,732,460 | `data/processed/link_prediction/edges_aggregated.parquet` |
| Unknown NT count | 19,658 | Calculated from neuron table |
| Unknown NT % | 14.1% | Calculated from neuron table |

---

## 6. Unverified Claims

| Claim | Status | Reason |
|-------|--------|--------|
| Majority baseline Accuracy/F1 | UNVERIFIED | Not stored in authoritative result files |
| Logistic Regression Accuracy/F1 | UNVERIFIED | Not stored in authoritative result files |
| GraphSAGE training time (306s) | UNVERIFIED | From `gnn_results.json` but not cross-validated |

All other major quantitative claims verified against authoritative artifacts.

---

## 7. Test Status

```
Collected: 131 tests
Passed: 131 tests (116 fast + 15 slow)
Failed: 0
Skipped: 0
```

---

## 8. Publication Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research report | `docs/FlyMind_Final_Research_Report.md` | ✅ Verified |
| Results traceability | `docs/RESULTS_TRACEABILITY.md` | ✅ Verified |
| Publication freeze | `docs/FINAL_PUBLICATION_FREEZE.md` | ✅ Created |
| Final results | `docs/FINAL_RESULTS.md` | ✅ Verified |
| Research audit | `docs/FINAL_RESEARCH_AUDIT.md` | ✅ Verified |
| Model artifact | `models/link_prediction_rf.pkl` | ✅ Present |
| Dashboard | `app/` | ✅ Functional |
| Presentation figures | `results/presentation/` | ✅ 6 figures |
| Test suite | `tests/` | ✅ 131 tests passing |

---

## 9. Scientific Freeze

> No further experimental changes are required for the current research version.

**Frozen:**
- All model architectures and hyperparameters
- All evaluation protocols and splits
- All reported metrics and results
- Feature definitions and engineering
- Negative sampling methodology

**Allowed after freeze:**
- Typo corrections
- Formatting improvements
- Documentation clarifications
- Presentation/UI improvements
- Packaging and deployment
- README updates
- Citation corrections

---

## 10. Git

```
Branch: docs/final-report-freeze
Commit: c1911d8
Message: docs: finalize research report and publication traceability
Files changed: 12 (227 insertions, 31 deletions)
Working tree: Clean
Main branch modified: NO
```

---

## 11. Remaining Limitations

1. **Dataset-specific:** Results apply to FlyWire FAFB dataset only
2. **Incomplete NT annotations:** 14.1% of neurons lack neurotransmitter type labels
3. **No biological validation:** Candidate connections require experimental confirmation
4. **Ranking, not probability:** Scores are model-derived rankings, not calibrated probabilities
5. **Experiment 2C aborted:** Neural network ablation experiment was not completed
6. **Raw data external:** Source dataset lives outside the repository
7. **No pinned dependencies:** requirements.txt does not specify exact versions

---

## 12. Remaining Work

After Phase 7, the following could be done:

1. **Presentation deck** — Create slides using `results/presentation/` figures
2. **PDF generation** — Convert Markdown to PDF if needed
3. **Dependency pinning** — Add exact versions to requirements.txt
4. **Path configurability** — Add environment variable overrides for raw data path
5. **Release preparation** — Tag releases, create CHANGELOG
6. **GitHub release** — Package for public distribution

---

*Phase 7 complete. Research results frozen. Publication-ready.*
