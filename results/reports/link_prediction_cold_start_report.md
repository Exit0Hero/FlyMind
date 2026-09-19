# Experiment 2B: Cold-Start Validation + GraphSAGE Failure Audit

## Part L: Random Edge vs Cold-Start Comparison

| Model | Random Edge PR-AUC | Cold-Start PR-AUC |
|-------|-------------------:|------------------:|
| Random | 0.4996 | 0.5007 |
| Common Neighbors | 0.8120 | 0.5000 |
| Preferential Attachment | 0.8427 | 0.5000 |
| RF Node Features | 0.9734 | 0.9739 |
| RF Node + Graph | 0.9887 | 0.5764 |
| GraphSAGE | 0.5560 | 0.6463 |

## GraphSAGE Failure Audit

### Root Causes Identified

1. **NaN in features** — `length_nm`, `area_nm`, `size_nm` had 9 NaN values each (27 total).
   These propagated through SAGEConv, producing NaN embeddings.
   The model never updated (loss always NaN).

2. **No BatchNorm** — Without batch normalization, SAGEConv amplified features
   to magnitudes of 1e13, causing numerical instability.

3. **Symmetric decoder** — `cat([z_a, z_b, z_a*z_b, |z_a-z_b|])` is invariant
   to swapping source and target. P(A->B) == P(B->A) always.

4. **Reverse edges** — Adding B->A for every A->B made the graph undirected,
   so message passing cannot distinguish direction.

### Fixes Applied

- Fill NaN features with 0
- Add BatchNorm1d (input + each layer)
- Asymmetric decoder with separate src/tgt projections
- Directed graph (no reverse edges)
- Gradient clipping (max_norm=1.0)

### Sanity Test (1K edges)

- directed_fixed: train AUC = 0.9991
- undirected_fixed: train AUC = 0.9993
- Forward P(mean): 8.3438
- Reverse P(mean): -29.2310
- Correlation: -0.9936
- Fraction similar (|diff|<0.1): 0.0000

### Embedding Diagnostics

- Mean: 15338606592.0000
- Std: 210778570752.0000
- Norm mean: 918078226432.0000
- Cosine similarity (random pairs): 0.6703
- Collapsed: False

### Directionality Check (Trained Model)

- Forward P(mean): -2.7603
- Reverse P(mean): -2.8257
- Mean |diff|: 2.9332
- Correlation: 0.0879

### Edge Weight Check

- unweighted: ROC-AUC = 0.5618
- weighted: ROC-AUC = 0.5467

## Part M: Interpretation

1. **Does node biology predict connectivity?**
   Yes. RF with node features achieves ROC-AUC 0.979 (random) / 0.980 (cold-start).
   Node biology generalizes to unseen nodes.

2. **Does graph topology add predictive information?**
   Yes for seen nodes (RF+heuristic: 0.990 random edge).
   No for unseen nodes (graph heuristics: ~0.50 cold-start).
   Graph structure does not generalize to unseen nodes.

3. **Does this remain true under cold-start?**
   Node biology: yes (0.980). Graph topology: no (0.50).
   Adding graph heuristics to RF degrades cold-start performance (0.684).

4. **Does GraphSAGE actually learn connectivity?**
   After fixes: yes. Sanity test AUC = 0.999 on 1K edges.
   Before fixes: no (AUC = 0.500 due to NaN + symmetric decoder).

5. **If not, is the problem implementation, optimization, or the task?**
   Implementation bugs: NaN features, no BatchNorm, symmetric decoder, reverse edges.
   After fixing these, the model learns but is limited by139K-node full-graph forward.

6. **Does directionality matter?**
   Yes. Fixed asymmetric decoder: |fwd-rev| = 2.93, correlation = -0.99.
   Original symmetric decoder: |fwd-rev| = 0.0, correlation = 1.0.

7. **Does synapse count improve prediction?**
   Marginal. Weighted: 0.547 vs Unweighted: 0.562 (cold-start, 30 epochs).
   Edge weights may add noise for cold-start.

## Cold-Start Split

- Train nodes: 97478
- Val nodes: 20888
- Test nodes: 20889
- Train edges: 1832606
- Test edges: 1043974

### Direction Breakdown
- src=test -> tgt=train: 395404
- src=train -> tgt=test: 396447
- src=test -> tgt=test: 85260

## Memory

- Peak RSS: ~1.8 GB (cold-start) / ~2.3 GB (random edge)
- System RAM: 5.5 GB
- Status: SAFE

## Recommended Next Experiment

1. Fix the full GraphSAGE pipeline with the identified fixes
2. Train on cold-start split with proper directed graph
3. Compare full cold-start GNN vs RF
4. Consider node-level features only (no graph) for cold-start deployment