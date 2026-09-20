# FlyMind Presentation Data Layer

## Overview

The Presentation Data Layer provides efficient, read-only access to FlyMind data for a future interactive dashboard. It wraps the Phase 2 inference layer and adds presentation-specific functionality.

**Critical:** All outputs preserve the distinction between:
- **OBSERVED CONNECTIONS** (from the connectome)
- **MODEL-SUGGESTED CANDIDATES** (hypotheses from the RF model)

---

## 1. Architecture

```
Processed Data
      |
      +-- Neuron Table (parquet)
      |         |
      +-- Edges (parquet)
      |         |
      +-- Inference Layer (RF model, features)
      |         |
      v         v
  FlyMindDataStore
      |
      +-- Neuron Lookup
      +-- Connectivity Lookup
      +-- Candidate Ranking
      +-- Experiment Results
      +-- Visualization Data
      |
      v
  Future Dashboard
```

---

## 2. Data Sources

| Source | Path | Description |
|--------|------|-------------|
| Neuron table | `data/processed/neuron_table.parquet` | 139,255 neurons x 41 features |
| Aggregated edges | `data/processed/link_prediction/edges_aggregated.parquet` | 3,732,460 directed edges |
| RF model | `models/link_prediction_rf.pkl` | Trained classifier (185 MB) |
| Node features | `data/processed/link_prediction/X_features.npy` | 139K x 15 float32 |
| ID mapping | `data/processed/link_prediction/id_to_idx.json` | root_id -> index |
| Experiment results | `results/reports/*.json` | Verified metrics |

---

## 3. API Reference

### `FlyMindDataStore`

Main class for the presentation layer.

```python
from src.link_prediction.presentation import FlyMindDataStore

store = FlyMindDataStore()
```

#### Neuron Lookup

```python
# Get full neuron metadata
neuron = store.get_neuron(root_id)  # -> NeuronInfo

# Get connectivity summary
summary = store.get_neuron_summary(root_id)  # -> NeuronSummary
# Returns: outgoing_count, incoming_count, total_synapses_out/in
```

#### Connectivity Lookup

```python
# Outgoing connections (source -> target)
outgoing = store.get_outgoing_connections(root_id, limit=10)
# -> list[Connection] with weight, num_neuropils, target metadata

# Incoming connections (source -> target)
incoming = store.get_incoming_connections(root_id, limit=10)
# -> list[Connection] with weight, num_neuropils, source metadata

# Full summary with top connections
summary = store.get_connectivity_summary(root_id, top_k=5)
# -> ConnectivitySummary
```

#### Candidate Ranking

```python
# Model-suggested candidate targets (HYPOTHESES, not confirmed)
candidates = store.rank_candidate_targets(source_root_id, k=10)
# -> list[CandidateTarget] with rank, score, target metadata
```

#### Experiment Results

```python
# All verified experiment results
results = store.get_experiment_results()
# -> dict with keys: 2a, 2b, 2d, 3

# Model comparison for visualization
comparison = store.get_model_comparison()
# -> list[dict] with model, roc_auc, pr_auc

# Calibration curve data
cal = store.get_calibration_data()
# -> list[dict] with bin, mean_predicted, observed_rate

# Feature importance
fi = store.get_feature_importance()
# -> list[FeatureImportance] with feature, importance, group
```

#### Visualization Data

```python
# Network neighborhood for visualization
hood = store.get_neighborhood(
    root_id,
    direction="outgoing",  # "outgoing", "incoming", or "both"
    max_nodes=50,
    include_candidates=True,  # Add model-suggested candidates
    candidate_k=10,
)
# -> NeighborhoodData with nodes[] and edges[]

# Score distribution for histogram
sd = store.get_score_distribution()
# -> dict with mean, median, std, p90, p95, p99
```

---

## 4. Data Classes

### `NeuronInfo`
```python
root_id: int
nt_type: Optional[str]       # Neurotransmitter type
nt_type_score: Optional[float]
primary_type: Optional[str]  # Cell type label
super_class: Optional[str]   # High-level class
flow: Optional[str]          # Information flow
coord_x/y/z: Optional[float]
length_nm/area_nm/size_nm: Optional[float]
side: Optional[str]          # Brain hemisphere
name: Optional[str]          # Proofread name
```

### `Connection`
```python
source_root_id: int
target_root_id: int
weight: int                  # Synapse count
num_neuropils: int
target_nt_type: Optional[str]
target_super_class: Optional[str]
source_nt_type: Optional[str]
source_super_class: Optional[str]
```

### `CandidateTarget`
```python
rank: int
source_root_id: int
target_root_id: int
score: float                 # RF prediction (0-1)
target_nt_type: Optional[str]
target_super_class: Optional[str]
target_primary_type: Optional[str]
```

### `NeighborhoodData`
```python
nodes: list[dict]  # Each: {id, label, type, super_class, nt_type}
edges: list[dict]  # Each: {source, target, type, weight/score}
source_root_id: int
direction: str
```

---

## 5. Memory Strategy

| Component | Strategy |
|-----------|----------|
| Neuron metadata | Dict lookup via inference layer (~50 MB) |
| Edge adjacency | defaultdict of lists (~100 MB for 3.7M edges) |
| RF model | Loaded once via inference layer (~185 MB) |
| Node features | Loaded once via inference layer (~8 MB) |
| Experiment results | JSON files loaded once (~50 KB) |

Peak memory: ~400 MB (datastore + inference layer).

---

## 6. Scientific Interpretation

- All candidate connections are **hypotheses**, not discoveries
- Observed connections come from the connectome dataset
- The RF model is a **computational ranking system**
- 97.3% of top candidates involve neurons with unknown NT types (reflecting the 14.1% unknown NT rate across the full dataset)
- Scores represent relative likelihood, not biological probability

---

## 7. Known Limitations

1. **Structural connectome only** -- no behavioral or functional data
2. **Incomplete NT annotation** -- 19,658 of 139,255 neurons (14.1%) lack NT type labels
3. **Candidate sampling** -- not all pairs evaluated
4. **Cold-start tradeoff** -- test node neighborhoods hidden
5. **Ranking, not probability** -- scores are relative rankings
6. **Edge index in memory** -- ~100 MB for 3.7M edges

---

## 8. Future Dashboard Interface

The dashboard can call:

```python
store = FlyMindDataStore()

# Neuron page
neuron = store.get_neuron(root_id)
summary = store.get_connectivity_summary(root_id, top_k=10)

# Connectivity visualization
outgoing = store.get_outgoing_connections(root_id, limit=50)
incoming = store.get_incoming_connections(root_id, limit=50)

# Candidate exploration
candidates = store.rank_candidate_targets(root_id, k=20)

# Network visualization
hood = store.get_neighborhood(root_id, direction="both", max_nodes=30)

# Research results
results = store.get_experiment_results()
comparison = store.get_model_comparison()
fi = store.get_feature_importance()
```
