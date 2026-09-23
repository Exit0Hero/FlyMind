# Changelog

All notable changes to FlyMind will be documented in this file.

## [v1.0.0] - 2026-09-20

### Research

- Final research freeze and audit (Phase 1)
- Cold-start evaluation protocol validated
- Random Forest model validated (ROC-AUC 0.9800)
- GraphSAGE comparative evaluation completed
- Experiment 2C aborted due to memory constraints (not a scientific result)

### Models

- Random Forest (100 estimators, 60 pair features)
- GraphSAGE (comparative model)
- Model artifact: `models/link_prediction_rf.pkl`

### Evaluation

- Random-edge holdout (ROC-AUC 0.9905 with heuristics)
- Cold-start evaluation (ROC-AUC 0.9800)
- Candidate ranking (Recall@10 = 0.814)
- Calibration analysis (ECE = 0.041)
- Robustness analysis (Experiment 3)

### Dashboard

- Streamlit interactive dashboard
- 6 pages: Overview, Neuron Explorer, Connection Predictor, Candidate Ranking, Research Results, About
- Demo Mode with verified example neurons
- Scientifically accurate terminology throughout

### Documentation

- Final research report (`docs/FlyMind_Final_Research_Report.md`)
- Results traceability (`docs/RESULTS_TRACEABILITY.md`)
- Publication freeze report (`docs/FINAL_PUBLICATION_FREEZE.md`)
- Model card (`docs/MODEL_CARD.md`)
- Data card (`docs/DATA_CARD.md`)
- Quickstart guide (`docs/QUICKSTART.md`)
- Demo script (`docs/DEMO_SCRIPT.md`)
- Presentation outline (`docs/PRESENTATION_OUTLINE.md`)

### Testing

- 131 tests passing (116 fast + 15 slow)
- Test suites: inference, experiment_2d, experiment_3, data_integrity, pipeline, presentation, dashboard, link_prediction, cold_start

### Limitations

- Dataset-specific to FlyWire FAFB
- 14.1% of neurons lack neurotransmitter annotations
- Candidate connections require biological validation
- Scores are model rankings, not biological probabilities
- Experiment 2C was not completed
