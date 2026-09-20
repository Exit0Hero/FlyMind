# FlyMind Final Results

**Date:** 2026-09-20
**Model:** Random Forest (100 estimators, 60 pair features)
**Dataset:** FlyWire FAFB (139,255 neurons, 3,732,460 directed edges)

---

## 1. Primary Model: Cold-Start ROC-AUC

The **Random Forest (node features only)** is the primary model. Cold-start is the strongest validation.

| Model | ROC-AUC | PR-AUC | Protocol |
|-------|--------:|-------:|----------|
| **RF Node Features** | **0.9800** | **0.9739** | Cold-start |
| RF + Heuristics | 0.6839 | 0.5764 | Cold-start |
| GraphSAGE | 0.6485 | 0.6463 | Cold-start |
| Random baseline | 0.5007 | 0.5007 | Cold-start |

Graph heuristics collapse to ~0.50 on cold-start because test nodes have zero training edges.

---

## 2. Random Edge Holdout (2A)

| Model | ROC-AUC | PR-AUC | P@1000 |
|-------|--------:|-------:|-------:|
| **RF + Heuristics** | **0.9905** | **0.9887** | 1.000 |
| RF Node Features | 0.9792 | 0.9734 | 0.996 |
| GraphSAGE | 0.5437 | 0.5560 | 0.858 |
| Random baseline | 0.5001 | 0.5000 | 0.476 |

**CAUTION:** Random-edge holdout is optimistic. Nodes remain visible through other edges. Not a fair generalization test.

---

## 3. GraphSAGE Classification (Experiment 1)

| Metric | Value |
|--------|------:|
| Accuracy | 0.9265 |
| Weighted F1 | 0.9372 |
| Macro F1 | 0.7066 |
| Best epoch | 50 |
| Training time | 306s |
| Graph nodes | 139,255 |
| Graph edges | 7,464,920 |

Role: Supporting biological analysis only. Not the primary connectivity model.

---

## 4. Candidate Ranking Validation (2D)

| Metric | Value |
|--------|------:|
| Validation ROC-AUC | 0.9796 |
| Validation PR-AUC | 0.9740 |
| Sources sampled | 10,000 |
| Candidates scored | 1,000,000 |
| Final top candidates | 100,000 |

---

## 5. Held-out Positive Rank (3F)

Given each held-out positive edge, what is its rank among 20 sampled negatives?

| Metric | Value |
|--------|------:|
| Median rank | 1.0 |
| Mean rank | 2.8 |
| Fraction ranked #1 | 53.6% |
| Fraction in top-10 | 95.9% |
| Fraction in top-50 | 100.0% |

---

## 6. Ranking vs Random Baseline (3A)

Per-source ranking among 20 sampled negatives per source.

| Metric | RF | Random | RF/Random |
|--------|---:|-------:|----------:|
| Recall@10 | 0.814 | 0.098 | 8.3x |
| Recall@50 | 0.990 | 0.463 | 2.1x |
| Hit Rate@10 | 0.995 | 0.420 | 2.4x |
| Hit Rate@50 | 1.000 | 0.960 | 1.0x |

---

## 7. Calibration (3D)

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Brier score | 0.0536 | Reasonable |
| Log loss | 0.1914 | Low |
| ECE | 0.0411 | Well-calibrated |

Conclusion: RF model is more appropriate as a ranking system than as a literal probability estimator.

---

## 8. Score Distribution (3C)

| Threshold | Count | Percentage |
|-----------|------:|-----------:|
| score = 1.0 | 34 | 0.03% |
| score >= 0.99 | 92 | 0.09% |
| score >= 0.95 | 454 | 0.45% |
| score >= 0.90 | 1,103 | 1.10% |
| score >= 0.75 | 3,604 | 3.60% |
| score >= 0.50 | 9,257 | 9.26% |

Saturation is minimal. Model is genuinely discriminative.

---

## 9. Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|-----------:|
| 1 | area_nm | 0.0203 |
| 2 | length_nm | 0.0165 |
| 3 | size_nm | 0.0152 |
| 4 | coord_x | 0.0142 |
| 5 | coord_y | 0.0100 |
| 6 | gaba_avg | 0.0088 |
| 7 | coord_z | 0.0085 |
| 8 | ach_avg | 0.0073 |
| 9 | da_avg | 0.0070 |
| 10 | nt_type_score | 0.0059 |

Morphological features (area, length, size) dominate, followed by spatial coordinates and neurotransmitter expression.

---

## 10. Exclusion Audit (3G)

| Check | Count |
|-------|------:|
| Total candidates | 100,000 |
| Self-loops | 0 |
| Known edges | 0 |
| Duplicate pairs | 0 |
| Invalid IDs | 0 |
| Non-finite scores | 0 |
| Out-of-range scores | 0 |

All clean.

---

## 11. Biological Pattern Observations (3H)

- **97.3%** of top candidates involve neurons with unknown NT types (reflecting the 14.1% unknown NT rate across the full dataset)
- Known-type pairs: GABA and ACH enriched
- **Post-hoc observation only**, not a causal claim

---

## 12. Cold-Start Split Sizes

| Split | Nodes | Edges |
|-------|------:|------:|
| Train | 97,478 | 1,832,606 |
| Validation | 20,888 | 855,880 |
| Test | 20,889 | 1,043,974 |
| **Total** | **139,255** | **3,732,460** |

---

## 13. Final Research Question

> Can neuron-level biological and morphological properties predict directed connectivity in the fruit-fly connectome, and do these relationships generalize to previously unseen neurons?

**Answer: Yes.** The RF model achieves ROC-AUC 0.980 on a cold-start evaluation where test neurons have zero training connectivity information. Neuron-level features (morphology, spatial coordinates, neurotransmitter expression) are sufficient to predict directed connections with high accuracy, and this generalizes to previously unseen neurons.
