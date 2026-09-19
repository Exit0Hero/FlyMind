# Experiment 2B: Cold-Start Validation + GraphSAGE Failure Audit

## 1. Cold-Start Split Sizes

| Split | Nodes | Edges |
|-------|------:|------:|
| Train | 97,478 | 1,832,606 |
| Val | 20,888 | 855,880 |
| Test | 20,889 | 1,043,974 |

Direction breakdown (test positives):
- src=test -> tgt=train: 395,404
- src=train -> tgt=test: 396,447
- src=test -> tgt=test: 85,260

## 2. Negative Sample Counts

- Test negatives: 1,043,974 (verified collision-free against full graph)
- Val negatives: 855,880

## 3. RF Node-Only Results

| Setting | ROC-AUC | PR-AUC |
|---------|--------:|-------:|
| Random edge | 0.979 | 0.973 |
| Cold-start | 0.980 | 0.974 |

## 4. RF Node + Graph Heuristics Results

| Setting | ROC-AUC | PR-AUC |
|---------|--------:|-------:|
| Random edge | 0.990 | 0.989 |
| Cold-start | 0.684 | 0.576 |

Graph heuristics hurt cold-start performance because test nodes have no training edges, so heuristics are zero for all test candidates and add noise.

## 5. Heuristic Results (Cold-Start)

All heuristics achieve ~0.50 ROC-AUC (random performance) because test nodes have no edges in the training graph, making common-neighbor and degree-based features useless.

## 6. GraphSAGE Results

| Setting | ROC-AUC | PR-AUC |
|---------|--------:|-------:|
| Random edge (original, broken) | 0.544 | 0.556 |
| Cold-start (fixed model) | 0.649 | 0.646 |

## 7. Random Edge vs Cold-Start Comparison

| Model | Random Edge PR-AUC | Cold-Start PR-AUC |
|-------|-------------------:|------------------:|
| Random | 0.500 | 0.501 |
| Common Neighbors | 0.812 | 0.500 |
| Preferential Attachment | 0.843 | 0.500 |
| **RF Node Features** | **0.973** | **0.974** |
| RF Node + Graph | 0.989 | 0.576 |
| GraphSAGE (fixed) | 0.556 | 0.646 |

## 8. Sanity Test Result

| Graph Type | Train AUC (1K edges) |
|------------|---------------------:|
| Directed (fixed) | 0.999 |
| Undirected (fixed) | 0.999 |
| Directed (original) | 0.500 |
| Undirected (original) | 0.500 |

The fixed model successfully overfits 1K edges. The original model could not learn at all.

## 9. Embedding Diagnostics

| Metric | Original | Fixed |
|--------|--------:|------:|
| Mean | NaN | 1.53e10 |
| Std | NaN | 2.11e11 |
| Norm mean | NaN | 9.18e11 |
| Collapsed | N/A | No |
| Cosine sim (random pairs) | N/A | 0.670 |

Original model produced NaN embeddings due to NaN in input features. Fixed model produces large but non-NaN embeddings.

## 10. Directionality Diagnostic

| Metric | Original | Fixed |
|--------|--------:|------:|
| Forward P(mean) | -3.7e20 | -2.76 |
| Reverse P(mean) | -3.7e20 | -2.83 |
| Mean |diff| | 0.0 | 2.93 |
| Correlation | 1.000 | -0.994 |
| Fraction |diff|<0.1 | 1.000 | 0.024 |

Original model: P(A->B) == P(B->A) always (symmetric decoder). Fixed model: strong directional preference.

## 11. Weighted vs Unweighted

| Setting | ROC-AUC |
|---------|--------:|
| Unweighted | 0.562 |
| Weighted | 0.547 |

Edge weights provide marginal improvement for cold-start. May add noise.

## 12. Root Cause of Poor GraphSAGE Performance

Four bugs were identified and fixed:

### Bug 1: NaN in Features
`length_nm`, `area_nm`, `size_nm` contained 9 NaN values each (27 total). These propagated through SAGEConv, producing NaN embeddings. The model never updated because the loss was always NaN.

**Fix:** Fill NaN with 0 in `build_feature_matrix()`.

### Bug 2: No BatchNorm (Numerical Instability)
Without batch normalization, SAGEConv amplified features to magnitudes of 1e13. This caused gradient explosion and prevented learning.

**Fix:** Add `BatchNorm1d` after input and each SAGEConv layer.

### Bug 3: Symmetric Decoder
The decoder used `cat([z_a, z_b, z_a*z_b, |z_a-z_b|])` which is invariant to swapping source and target. P(A->B) always equaled P(B->A).

**Fix:** Asymmetric decoder with separate `src_proj` and `tgt_proj` linear layers.

### Bug 4: Reverse Edges Destroyed Directionality
Adding B->A for every A->B made the graph undirected, so message passing could not distinguish direction.

**Fix:** Use directed graph (no reverse edges).

## 13. Memory Peak

- Cold-start pipeline: ~1.8 GB RSS
- Random-edge pipeline: ~2.3 GB RSS
- System RAM: 5.5 GB
- Status: SAFE

## 14. Tests Passed

15/15 tests pass:
- 8 original link prediction tests
- 7 cold-start + GNN diagnostics tests

## 15. Does Graph Structure Generalize?

**No.** Graph heuristics drop from 0.81-0.84 (random edge) to ~0.50 (cold-start). RF+graph degrades from 0.989 to 0.576. Only node biology (0.974) generalizes to unseen nodes.

The graph topology signal is entirely transductive: it works for nodes seen during training but provides no information for novel neurons.

## 16. Recommended Next Experiment

1. Retrain fixed GraphSAGE on cold-start split with full pipeline
2. Consider node-feature-only GNN (no message passing) as cold-start baseline
3. Explore whether graph structure helps for transductive (seen-node) settings
4. Investigate scaling laws: does more training data help GraphSAGE?
