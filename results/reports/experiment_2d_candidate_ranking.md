# Experiment 2D: Explainable Candidate Connection Ranking

## Objective

Use the validated RF node-feature model (ROC-AUC 0.980) to rank plausible candidate directed connections between neurons in the FlyMind connectome. This produces scientifically conservative, model-suggested hypotheses for validation, without requiring neural network training.

## Design

### Why This Approach

- **No all-pairs enumeration**: 139K nodes x 139K = 19.4B pairs is impossible. Instead: 10K random sources x 100 random targets = 1M candidates (0.005% coverage).
- **No neural network**: Experiment 2C failed due to RAM constraints (~1 GB RSS base + PyG overhead). The RF model (185 MB, 60 features) is memory-efficient and already validated.
- **Explainability**: RF feature importances provide per-candidate explanations. Feature importance groups quantify whether source biology, target biology, or their interaction drives predictions.
- **Conservative scope**: Only random-sampled candidates are ranked (not all-pairs). Top-ranked candidates represent plausible hypotheses, not predicted edges.

### Pipeline

1. Load: RF model, features (15 per node), edge index, cold-start test splits
2. Validate: Re-run ROC-AUC, PR-AUC, P@K on held-out cold-start edges
3. Generate candidates: For each source, randomly sample 100 targets (seed=42); exclude self-loops and known edges
4. Score: Chunked RF inference (5K pairs per batch); accumulate top-10 targets per source
5. Rank: Flatten all (source, target, score) triples; sort by score; take top-100K
6. Analyze: Feature importance, connection type analysis, post-hoc biological annotation

### Safety Constraints

- Never materializes 19.4B candidate matrix
- Chunked pair features (5K at a time) keep memory below 1.1 GB
- Negative sampling from valid node IDs only (no zero-padding)
- Verified: no self-loops, no known edges, no duplicates

## Results

### Model Validation (Held-out Cold-start Edges)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9796 |
| PR-AUC | 0.9740 |
| P@10 | 1.0000 |
| Hits@100 | 100 |

Consistent with Experiment 2B results.

### Candidate Generation

| Parameter | Value |
|-----------|-------|
| Sources sampled | 10,000 |
| Targets per source | 100 |
| Total candidates scored | 1,000,000 |
| Final top candidates | 100,000 |

### Score Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 0.1346 |
| Median | 0.0300 |
| Std | 0.2207 |
| P90 | 0.48 |
| P95 | 0.68 |
| P99 | 0.91 |

Most candidates receive low scores (median 0.03), with a small fraction receiving high confidence (P99 = 0.91). This bimodal distribution is expected: most random pairs are truly non-connected, while a few high-scoring pairs represent plausible connections.

### Feature Importance

**Top-10 Individual Features:**

| Rank | Feature | Importance |
|------|---------|------------|
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

**Feature Group Importances:**

| Group | Importance | Interpretation |
|-------|------------|----------------|
| Source biology | 0.052 | Source neuron properties dominate predictions |
| Target biology | 0.033 | Target neuron properties contribute substantially |
| Source-target interaction | 0.025 | Interaction terms (x_A * x_B) capture complementarity |
| Source-target difference | 0.023 | Similarity between source and target features |

### Top-20 Annotated Candidates

See `results/candidates/top20_annotated.csv` for the full top-20 with biological annotations (super_class, primary_type, NT type, flow, coordinates).

Notable patterns in top candidates:
- Most top candidates are same-NT-type connections (ACH -> ACH or GABA -> GABA)
- All are classified as intrinsic flow (within the same brain region system)
- Source and target neurons are often from similar neuropil regions (spatial proximity)

## Output Files

| File | Description |
|------|-------------|
| `results/candidates/top_candidate_connections.csv` | Top 100K candidate pairs with scores |
| `results/candidates/top20_annotated.csv` | Top 20 with biological annotations |
| `results/reports/experiment_2d_results.json` | Structured results (validation, distribution, importances) |
| `results/figures/experiment_2d/2d_score_distribution.png` | Score distribution + feature group importances |
| `results/figures/experiment_2d/2d_feature_importance.png` | Top-10 individual feature importances |
| `results/figures/experiment_2d/2d_score_heatmap.png` | Source x target score heatmap |

## Memory Profile

| Stage | RSS |
|-------|-----|
| Program start | 107 MB |
| After loading RF | 549 MB |
| After loading edges | 958 MB |
| After validation | 1031 MB |
| After candidate scoring | 1038 MB |
| Final | 1056 MB |

Peak: 1056 MB, well under 2 GB limit.

## Limitations

1. **Random sampling only**: Candidates are randomly sampled, not intelligently pruned. Many true connections may never appear as candidates.
2. **No edge features**: Only node features are used. Structural features (shared neuropils, path length) are not incorporated.
3. **High P@K on validation**: The model's perfect precision at small K on held-out edges suggests near-deterministic memorization of common connection patterns, not true generalization to novel connections.
4. **No biological validation**: Top candidates are computational hypotheses. Electrophysiological or connectomic validation is needed.
5. **Coverage**: 1M candidates covers 0.005% of all possible directed pairs. Missing candidates include potentially high-value connections.

## Next Steps

1. Biological validation: Cross-reference top candidates with published connectome data
2. Intelligent candidate pruning: Use neuropil-based heuristics to focus sampling on anatomically plausible pairs
3. Edge features: Incorporate shared-neuropil, spatial-proximity, and degree-based features
