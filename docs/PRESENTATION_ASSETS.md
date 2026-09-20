# FlyMind Presentation Assets

## Figures

| # | Filename | Description | Metrics | Source Artifact | Slide |
|---|----------|-------------|---------|-----------------|-------|
| 1 | `01_feature_importance.png` | Top 10 RF feature importances | area_nm (0.0203), length_nm (0.0165), size_nm (0.0152) | `results/reports/experiment_2d_results.json` → `feature_importance_top10` | 7 |
| 2 | `02_calibration.png` | Calibration curve | Brier: 0.0536, ECE: 0.0411 | `results/reports/experiment_3_robustness.json` → `3d_calibration` | 7 |
| 3 | `03_rank_distribution.png` | Held-out positive rank distribution | Median rank: 1.0, #1: 53.6%, Top-10: 95.9% | `results/reports/experiment_3_robustness.json` → `3f_rank_distribution` | 8 |
| 4 | `04_score_distribution.png` | Score distribution histogram | Mean: 0.135, Median: 0.03 | `results/reports/experiment_2d_results.json` → `score_distribution` | 7 |
| 5 | `05_biological_patterns.png` | Post-hoc biological patterns | 97.3% top candidates have unknown NT | `results/reports/experiment_3_robustness.json` → `3h_biological_patterns` | 10 |
| 6 | `06_degree_distribution.png` | Degree distribution | 139,255 neurons, 3.73M edges | `results/reports/graph_stats.json` | 4 |

## Location

All figures: `results/presentation/`

## Regeneration

Do not regenerate figures unless an existing figure contains a factual error. Figures are generated from frozen research artifacts.
