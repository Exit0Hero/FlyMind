# FlyMind Model Card

## Model

**Type:** Random Forest Classifier
**Framework:** scikit-learn
**Estimators:** 100
**File:** `models/link_prediction_rf.pkl`

## Intended Use

Research-oriented candidate ranking and analysis of directed neuron connectivity in the Drosophila connectome.

**Primary use cases:**
- Prioritizing pairs for experimental validation
- Analyzing relationships between neuron properties and connectivity
- Exploring connectome structure

## Not Intended For

- Clinical or medical decisions
- Autonomous biological conclusions
- Claiming confirmed biological connections
- Inferring causality
- Replacing experimental validation
- Predicting behavior or neural circuits

## Inputs

15 neuron-level features:

| # | Feature | Type | Category |
|---|---------|------|----------|
| 1 | `nt_type_score` | numeric | neurotransmitter |
| 2 | `da_avg` | numeric | neurotransmitter |
| 3 | `ser_avg` | numeric | neurotransmitter |
| 4 | `gaba_avg` | numeric | neurotransmitter |
| 5 | `glut_avg` | numeric | neurotransmitter |
| 6 | `ach_avg` | numeric | neurotransmitter |
| 7 | `oct_avg` | numeric | neurotransmitter |
| 8 | `length_nm` | numeric | morphology |
| 9 | `area_nm` | numeric | morphology |
| 10 | `size_nm` | numeric | morphology |
| 11 | `coord_x` | numeric | spatial |
| 12 | `coord_y` | numeric | spatial |
| 13 | `coord_z` | numeric | spatial |
| 14 | `flow` | categorical | classification |
| 15 | `side_x` | categorical | spatial |

## Pair Representation

For directed pair (A, B), 60-dimensional feature vector:

```
[x_A, x_B, |x_A - x_B|, x_A * x_B]
```

- Source features (15)
- Target features (15)
- Absolute difference (15)
- Element-wise product (15)

## Primary Evaluation

Cold-start directed link prediction.

Training edges touching held-out neurons are excluded, testing generalization to previously unseen neurons.

## Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9800 |
| PR-AUC | 0.9739 |
| Recall@10 | 0.814 |
| Hit Rate@10 | 0.995 |

## Limitations

- **Dataset-specific:** Trained on FlyWire FAFB dataset
- **Incomplete annotations:** 14.1% of neurons lack neurotransmitter type labels
- **Ranking, not probability:** Scores are model-derived rankings, not calibrated probabilities
- **No biological validation:** Candidate connections require experimental confirmation
- **Statistical, not causal:** Feature importance describes model behavior, not biological causation

## Citation

If you use this model, please cite the FlyMind project.
