# FlyMind Presentation Outline

**Duration:** 10–15 minutes
**Slides:** 10

---

## Slide 1 — Title

**FlyMind**

Predicting Directed Neuron Connectivity from Biological and Morphological Features in the Drosophila Connectome

---

## Slide 2 — The Problem

- Mapping neural connectivity is difficult
- Drosophila connectome: 139,255 neurons, 3.73M directed edges
- Can measurable neuron properties predict connections?
- Do relationships generalize to unseen neurons?

---

## Slide 3 — Research Question

> Can neuron-level biological and morphological properties predict directed connectivity in the fruit-fly connectome, and do these relationships generalize to previously unseen neurons?

---

## Slide 4 — Dataset

| Property | Value |
|----------|-------|
| Neurons | 139,255 |
| Directed edges | 3,732,460 |
| Node features | 15 |
| Source | FlyWire FAFB |

Features: neurotransmitter profiles, morphology, spatial coordinates, classification labels

---

## Slide 5 — Method

```
Neuron Features (15D)
     ↓
Pair Features (60D)
[x_A, x_B, |x_A-x_B|, x_A*x_B]
     ↓
Random Forest (100 estimators)
     ↓
Connection Score (0-1)
     ↓
Candidate Ranking
```

GraphSAGE evaluated as comparative model.

---

## Slide 6 — Evaluation

**Two protocols:**

1. **Random-edge holdout:** Optimistic—nodes visible through other edges
2. **Cold-start:** Held-out neurons have zero training edges—stricter generalization test

Cold-start is the primary evaluation.

---

## Slide 7 — Main Result

| Model | ROC-AUC | PR-AUC |
|-------|--------:|-------:|
| Random baseline | 0.5007 | — |
| **RF node features** | **0.9800** | **0.9739** |
| RF + heuristics | 0.6839 | 0.5764 |
| GraphSAGE | 0.6485 | 0.6463 |

Cold-start: RF node features retained strong performance.

---

## Slide 8 — Candidate Ranking

| Metric | Value |
|--------|-------|
| Sources evaluated | 10,000 |
| Candidates per source | 100 |
| Recall@10 | 0.814 |
| Hit Rate@10 | 0.995 |
| Improvement over random | 8.3x at K=10 |

Candidates are hypotheses, not confirmed connections.

---

## Slide 9 — Live Demo

**Show:** Streamlit Dashboard

1. Overview with key metrics
2. Neuron Explorer
3. Connection Predictor
4. Candidate Ranking
5. Research Results

---

## Slide 10 — Limitations + Future Work

**Limitations:**
- Dataset-specific (FlyWire FAFB)
- 14.1% neurons lack NT annotations
- No biological validation of candidates
- Scores are rankings, not probabilities

**Future work:**
- Richer graph models
- Improved annotations
- Experimental validation
- Larger-scale biological analysis

**Closing:** FlyMind provides computational prioritization, not biological proof.

---

## Presentation Assets

| Slide | Figure |
|-------|--------|
| 4 | `results/presentation/06_degree_distribution.png` |
| 7 | `results/presentation/01_feature_importance.png` |
| 7 | `results/presentation/02_calibration.png` |
| 8 | `results/presentation/03_rank_distribution.png` |
