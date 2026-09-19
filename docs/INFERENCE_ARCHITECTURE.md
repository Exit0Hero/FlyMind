# FlyMind Inference Architecture

## Overview

The FlyMind inference layer provides a clean backend interface for connectivity prediction between neurons in the FlyWire FAFB fruit-fly connectome.

**Critical:** All outputs are **model-suggested candidate connections** (hypotheses), NOT biological discoveries or confirmed connections.

---

## 1. Model Artifact

| Property | Value |
|----------|-------|
| File | `models/link_prediction_rf.pkl` |
| Type | `sklearn.ensemble.RandomForestClassifier` |
| Estimators | 100 |
| Pair feature dim | 60 (15 node features x 4) |
| Training | Random edge holdout with 1:1 negative sampling |
| Validation | Cold-start ROC-AUC 0.9796 |

The pickle contains:
```python
{
    "model": RandomForestClassifier,  # trained sklearn model
    "feature_cols": list[str],        # 15 base feature names
}
```

---

## 2. Input Features

### Node-Level Features (15)

| # | Feature | Type | Category |
|---|---------|------|----------|
| 1 | nt_type_score | numeric | neurotransmitter |
| 2 | da_avg | numeric | neurotransmitter |
| 3 | ser_avg | numeric | neurotransmitter |
| 4 | gaba_avg | numeric | neurotransmitter |
| 5 | glut_avg | numeric | neurotransmitter |
| 6 | ach_avg | numeric | neurotransmitter |
| 7 | oct_avg | numeric | neurotransmitter |
| 8 | length_nm | numeric | morphology |
| 9 | area_nm | numeric | morphology |
| 10 | size_nm | numeric | morphology |
| 11 | coord_x | numeric | spatial |
| 12 | coord_y | numeric | spatial |
| 13 | coord_z | numeric | spatial |
| 14 | flow | categorical | classification |
| 15 | side_x | categorical | spatial |

### Pair Feature Construction

For directed pair (A, B):
```
[x_A, x_B, |x_A - x_B|, x_A * x_B]
```

Result: 15 x 4 = **60 dimensions**

---

## 3. Architecture Diagram

### Single Pair Scoring

```
Neuron A ---+
           +--> Pair Feature Builder --> RF --> Score (0-1)
Neuron B ---+
```

### Candidate Ranking

```
Source Neuron
      |
Sample Candidate Pool
      |
Exclude Self-Loops
      |
Exclude Known Observed Edges
      |
Build Pair Features (chunked)
      |
RF Inference
      |
Sort by Score (descending)
      |
Return Top-K Candidates
```

---

## 4. API Reference

### `FlyMindInference`

Main class providing the inference interface.

```python
from src.link_prediction.inference import FlyMindInference

model = FlyMindInference()
```

#### `get_neuron(root_id: int) -> NeuronInfo`

Retrieve metadata for a single neuron.

**Returns:**
- `root_id`: Neuron identifier
- `nt_type`: Neurotransmitter type (if annotated)
- `nt_type_score`: Confidence of NT prediction
- `primary_type`: Cell type label
- `super_class`: High-level neuron class
- `flow`: Information flow direction
- `coord_x/y/z`: Body coordinates
- `length_nm/area_nm/size_nm`: Morphology measurements
- `side`: Brain hemisphere
- `name`: Proofread name

#### `score_connection(source_root_id, target_root_id) -> ConnectionScore`

Score a directed pair for connectivity likelihood.

**Returns:**
- `source_root_id`: Presynaptic neuron
- `target_root_id`: Postsynaptic neuron
- `score`: RF prediction (0-1, higher = more likely connection)
- `is_known_edge`: Whether this is an observed connection
- `is_self_loop`: Whether source == target

**Directionality:** `A -> B` is scored independently from `B -> A`.

#### `rank_candidate_targets(source_root_id, k=10, ...) -> list[CandidateTarget]`

Rank candidate target neurons for a given source.

**Parameters:**
- `source_root_id`: Presynaptic neuron to rank targets for
- `k`: Number of top candidates to return
- `candidate_pool_size`: Random candidates to sample (default 1000)
- `seed`: Random seed for reproducibility

**Returns:**
List of `CandidateTarget` objects with:
- `rank`: Position in ranking (1-indexed)
- `source_root_id`: Input source
- `target_root_id`: Candidate target
- `score`: RF prediction
- `target_nt_type`: Target's NT type (post-hoc annotation)
- `target_super_class`: Target's class (post-hoc annotation)
- `target_primary_type`: Target's cell type (post-hoc annotation)

---

## 5. Candidate Safety

The inference layer enforces these constraints:

| Constraint | Implementation |
|------------|----------------|
| No self-loops | Explicit filter: `cand_ids != source_root_id` |
| No known edges | Compact sorted pair array + binary search |
| No duplicate pairs | Random sampling with numpy (no replacement) |
| Valid neuron IDs | KeyError raised for unknown IDs |
| Finite scores | RF predict_proba always returns [0,1] |
| Memory-safe | Chunked feature construction, no all-pairs |

---

## 6. Memory Strategy

| Component | Strategy |
|-----------|----------|
| Node features | NumPy array (8.4 MB for 139K x 15) |
| Edge index | Compact uint64 sorted array (~30 MB for 3.7M edges) |
| Pair features | Chunked construction (10K pairs/chunk) |
| RF model | Single pickle load (~185 MB) |
| Neuron metadata | Dict lookup (lazy-loaded) |

Peak memory: ~500 MB (model + data + working set).

---

## 7. Scientific Interpretation

- The RF model is a **computational ranking system**
- Scores represent relative likelihood, not absolute biological probability
- Candidate connections are **hypotheses** requiring experimental validation
- The model generalizes to unseen neurons (cold-start validated)
- 97.3% of top candidates involve neurons with unknown NT types

---

## 8. Limitations

1. **Structural connectome only** -- no behavioral or functional data
2. **Incomplete NT annotation** -- 19K of 139K neurons lack NT labels
3. **Candidate sampling** -- not all pairs evaluated, sampling-based approach
4. **Cold-start tradeoff** -- test node neighborhoods hidden, reducing heuristic signal
5. **Ranking, not probability** -- scores are relative rankings, not calibrated probabilities
