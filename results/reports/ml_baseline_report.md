# FlyMind ML Baseline Report

**Date:** 2026-09-19
**Task:** Neuron `super_class` classification using the FlyWire FAFB connectome

---

## 1. Dataset

| Metric | Value |
|--------|-------|
| Total neurons | 139,255 |
| Filtered edges (Princeton) | 5,342,446 |
| Aggregated directed edges | 3,732,460 |
| Self-loops | 0 |
| Duplicate edges (after aggregation) | 0 |

---

## 2. Target

**`super_class`** — a high-level biological classification of neuron type.

| Class | Count | % |
|-------|-------|---|
| optic | 77,873 | 55.9% |
| central | 32,381 | 23.3% |
| sensory | 16,938 | 12.2% |
| visual_projection | 7,684 | 5.5% |
| ascending | 1,750 | 1.3% |
| descending | 1,305 | 0.9% |
| sensory_ascending | 612 | 0.4% |
| visual_centrifugal | 522 | 0.4% |
| motor | 110 | 0.1% |
| endocrine | 80 | 0.1% |

**10 classes. Significant class imbalance** — `optic` is 55.9% while `endocrine` is 0.1%.

---

## 3. Features

### Safe candidate features (15 total)

**Numeric (13):**
- `nt_type_score` — neurotransmitter prediction confidence
- `da_avg`, `ser_avg`, `gaba_avg`, `glut_avg`, `ach_avg`, `oct_avg` — neurotransmitter probability scores
- `length_nm`, `area_nm`, `size_nm` — morphological measurements
- `coord_x`, `coord_y`, `coord_z` — spatial position

**Categorical (2):**
- `flow` — intrinsic/afferent/efferent (3 values)
- `side_x` — left/right (3 values)

---

## 4. Leakage Audit

### Excluded features and reasons

| Feature | Reason for exclusion |
|---------|---------------------|
| `primary_type` | Directly derives from super_class hierarchy |
| `class` | Child category of super_class — trivially reveals target |
| `sub_class` | Child category of super_class — trivially reveals target |
| `name` | Proofread name encodes cell type identity |
| `connectivity_tags` | Community labels derived from connectivity patterns |
| `processed_labels` | Community-refined labels — may encode class |
| `nt_type` | Dominant neurotransmitter — may be derived from class |
| `hemilineage` | Developmental lineage — high correlation with class |
| `nerve` | Nerve assignment — low coverage (6.9%), may encode class |
| `group_x`, `group_y` | Neuron group — may encode class |
| `type_x`, `family`, `subsystem`, `category`, `side_y` | Visual neuron annotations — may encode class |
| `hemisphere`, `type_y`, `column_id`, `x`, `y`, `p`, `q` | Column assignment — low coverage (32.7%), may encode class |

**Decision:** Only the 15 safe features listed above are used. No features that encode or trivially reveal `super_class` are included.

---

## 5. Splitting

| Split | Nodes | % |
|-------|-------|---|
| Train | 97,477 | 70.0% |
| Validation | 20,889 | 15.0% |
| Test | 20,889 | 15.0% |

**Method:** Stratified split by `super_class` to preserve class proportions.
**Seed:** 42 (fixed, reproducible).
**Preprocessing statistics** (imputation, scaling, encoding) fitted **only on training data**.

---

## 6. Baselines

| Model | Features | Graph | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|-------|----------|----------|-------------|
| Majority class | — | No | 0.5592 | 0.0717 | 0.3127 |
| Logistic Regression | Bio (15) | No | 0.8144 | 0.6301 | 0.8058 |
| Random Forest | Bio (15) | No | 0.9727 | 0.8724 | 0.9726 |
| Random Forest | Graph (7) | No | 0.6655 | 0.3996 | 0.6144 |
| Random Forest | Bio (15) | No | 0.9727 | 0.8724 | 0.9726 |
| Random Forest | Bio+Graph (22) | No | 0.9773 | 0.9005 | 0.9772 |
| GraphSAGE | Node features (15) | **Yes** | 0.9265 | 0.7066 | 0.9372 |

---

## 7. GNN Architecture

```
GraphSAGE:
  Input: 15 features
  → SAGEConv(15, 64) → LayerNorm → ReLU → Dropout(0.3)
  → SAGEConv(64, 64) → LayerNorm → ReLU → Dropout(0.3)
  → SAGEConv(64, 64)

Classifier:
  Linear(64, 10) → super_class probabilities
```

**Configuration:**
- Hidden dim: 64
- Layers: 2
- Dropout: 0.3
- Optimizer: Adam (lr=0.005, weight_decay=1e-4)
- Class weights: inverse frequency, normalized
- Early stopping patience: 10
- Training: full-batch on CPU

**Graph construction:**
- Reverse edges added for message passing (7,464,920 total edges)
- Edge weights: aggregated synapse counts

---

## 8. Training

| Metric | Value |
|--------|-------|
| Best epoch | 50 |
| Training time | 306.4 seconds (~5.1 minutes) |
| Final train loss | 0.2761 |
| Best val accuracy | 0.9259 |
| Best val macro F1 | 0.7007 |

---

## 9. Test Results

### Overall

| Metric | Value |
|--------|-------|
| Accuracy | 0.9265 |
| Macro F1 | 0.7066 |
| Weighted F1 | 0.9372 |

### Per-class (GraphSAGE)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| ascending | 0.97 | 0.86 | 0.91 | 263 |
| central | 0.99 | 0.91 | 0.95 | 4,857 |
| descending | 0.54 | 0.91 | 0.68 | 196 |
| endocrine | 0.35 | 0.67 | 0.46 | 12 |
| motor | 0.20 | 0.94 | 0.33 | 16 |
| optic | 0.99 | 0.93 | 0.96 | 11,681 |
| sensory | 1.00 | 1.00 | 1.00 | 2,541 |
| sensory_ascending | 0.74 | 0.96 | 0.83 | 92 |
| visual_centrifugal | 0.18 | 0.86 | 0.29 | 78 |
| visual_projection | 0.55 | 0.82 | 0.66 | 1,153 |

---

## 10. Ablation Summary

| Model | Features | Graph | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|-------|----------|----------|-------------|
| Majority class | — | No | 0.5592 | 0.0717 | 0.3127 |
| Logistic Regression | Bio (15) | No | 0.8144 | 0.6301 | 0.8058 |
| Random Forest | Bio (15) | No | 0.9727 | 0.8724 | 0.9726 |
| Random Forest | Graph (7) | No | 0.6655 | 0.3996 | 0.6144 |
| Random Forest | Bio+Graph (22) | No | **0.9773** | **0.9005** | **0.9772** |
| GraphSAGE | Node features | Yes | 0.9265 | 0.7066 | 0.9372 |

### Key findings

1. **Graph features alone are weak** — RF with graph features only gets Macro F1 0.40, much worse than bio features alone (0.87).
2. **Graph features help when combined with bio features** — RF bio+graph (0.9005) beats RF bio-only (0.8724) by +2.8% on Macro F1.
3. **GNN underperforms traditional ML** — GraphSAGE (0.7066) is significantly worse than Random Forest (0.8724). This is expected for node classification with rich node features, where neighborhood aggregation may dilute informative features.
4. **The bio features are very strong** — 15 features achieving 0.97 accuracy with Random Forest suggests the biological annotations already capture most of the signal for super_class.

---

## 11. Interpretation

The model appears to use primarily:

- **Neurotransmitter scores** — strong signal for distinguishing optic (glutamate-dominated) from sensory (ach-dominated) neurons
- **Morphological features** — length, area, size differentiate neuron types
- **Spatial coordinates** — neurons in similar brain regions tend to share super_class

The GNN's lower performance suggests that in this connectome, **node features carry more discriminative information than graph neighborhood** for super_class prediction. This makes biological sense: super_class is defined by cell biology, not connectivity.

---

## 12. Limitations

1. **The GNN is a simple 2-layer GraphSAGE** — more expressive architectures (GAT, Graph Transformer) may perform better.
2. **Full-batch training on CPU** — no mini-batch neighbor sampling (torch-sparse unavailable for Python 3.14).
3. **Class imbalance not fully addressed** — class weighting helps but rare classes (endocrine, motor) still have low F1.
4. **No edge features in GNN** — neuropil-specific connection types are not used.
5. **Super_class may not be the most interesting target** — finer-grained types (class, sub_class) would be harder.
6. **Graph is undirected in practice** — reverse edges are added for message passing, which may not reflect biological directionality.

---

## 13. Generated Files

| File | Description |
|------|-------------|
| `models/graphsage_superclass_best.pt` | Best model checkpoint |
| `models/preprocessor_config.json` | Preprocessing configuration |
| `results/figures/class_distribution.png` | Target class distribution |
| `results/figures/training_curves.png` | Loss/accuracy/F1 curves |
| `results/figures/confusion_matrix_gnn.png` | GNN confusion matrix |
| `results/reports/local_explanations.txt` | 5 neuron neighborhood analyses |
| `results/reports/gnn_results.json` | GNN metrics JSON |

---

## 14. Next Experiments

### Immediately feasible

1. **GNN with edge features** — incorporate neuropil and NT type as edge attributes
2. **GAT (Graph Attention)** — attention may help weight neighbor importance
3. **Harder target: `class` or `sub_class`** — more fine-grained, more challenging
4. **Graph features as additional GNN input** — concatenate degree/weight features with node features
5. **Ensemble: RF predictions + GNN predictions** — combine strengths of both approaches

### Worth investigating

6. **Link prediction as auxiliary task** — multi-task learning to improve node representations
7. **Neuropil-aware model** — separate models for different brain regions
8. **Active learning** — identify neurons where the model is most uncertain
9. **Contrastive learning** — pre-train on graph structure, fine-tune on classification

### Not recommended yet

10. ~~Behavior prediction~~ — no behavioral labels in this dataset
11. ~~Circuit discovery~~ — needs more rigorous evaluation framework
12. ~~3D visualization~~ — premature before model is validated

---

## 15. Research Integrity

This model learns **statistical patterns** between biological features, morphological properties, and neuron class labels from the FlyWire connectome. It does not:

- Understand fly brain function
- Discover biological mechanisms
- Predict fly behavior
- Imply causality from correlation

The high accuracy reflects that **super_class is well-characterized** and correlates strongly with neurotransmitter profiles and morphology. The model is a useful baseline, not a neuroscience discovery.
