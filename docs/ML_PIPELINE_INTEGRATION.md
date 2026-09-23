# ML Pipeline Integration

## Overview

FlyMind's ML pipeline is fully integrated into the full-stack architecture.

## Pipeline Stages

```
FlyWire Data (dataset/*.csv.gz)
      ↓
ETL (src/build_neuron_table.py)
      ↓
Processed Data (data/processed/neuron_table.parquet)
      ↓
Feature Engineering (src/link_prediction/features.py)
      ↓
Random Forest (models/link_prediction_rf.pkl)
      ↓
Evaluation (results/reports/*.json)
      ↓
Prediction (backend/app/services/ml_service.py)
      ↓
Candidate Ranking (backend/app/api/candidates.py)
```

## Component Locations

| Component | Location | Description |
|---|---|---|
| Production Model | `models/link_prediction_rf.pkl` | Random Forest (100 estimators, 60D features) |
| Model Loader | `src/link_prediction/inference.py` | `FlyMindInference` class |
| Feature Contract | `src/link_prediction/features.py` | `build_pair_features()` - 15 node features → 60D pair features |
| Prediction | `src/link_prediction/inference.py:score_connection()` | Score a directed neuron pair |
| Candidate Ranking | `src/link_prediction/inference.py:rank_candidate_targets()` | Rank candidate targets for a source |
| Neuron Lookup | `src/link_prediction/inference.py:get_neuron()` | Get neuron metadata |
| Processed Data | `data/processed/neuron_table.parquet` | 139,255 neurons with metadata |
| Edge Data | `data/processed/link_prediction/edges_aggregated.parquet` | Directed edges with weights |
| Node Features | `data/processed/link_prediction/X_features.npy` | (139255, 15) float32 matrix |
| ID Mapping | `data/processed/link_prediction/id_to_idx.json` | root_id → matrix index |
| Evaluation Reports | `results/reports/*.json` | 8 experiment result files |

## Feature Contract

### Node Features (15 dimensions)

1. `nt_type_score` - Neurotransmitter type confidence score
2. `da_avg` - Dopamine average
3. `ser_avg` - Serotonin average
4. `gaba_avg` - GABA average
5. `glut_avg` - Glutamate average
6. `ach_avg` - Acetylcholine average
7. `oct_avg` - Octopamine average
8. `length_nm` - Neuron length (nm)
9. `area_nm` - Neuron surface area (nm²)
10. `size_nm` - Neuron volume (nm³)
11. `coord_x` - X coordinate
12. `coord_y` - Y coordinate
13. `coord_z` - Z coordinate
14. `flow_enc` - Flow direction (encoded)
15. `side_x_enc` - Hemisphere side (encoded)

### Pair Features (60 dimensions)

For neurons A and B: `[A, B, |A-B|, A*B]`

- 15 features from neuron A
- 15 features from neuron B
- 15 absolute differences
- 15 element-wise products

## Validated Metrics

### Cold-Start Split (Production Model)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9800 |
| PR-AUC | 0.9739 |
| Recall@10 | 0.814 |
| Hit Rate@10 | 0.995 |

### Standard Split

| Metric | Value |
|---|---|
| ROC-AUC | 0.979 |
| PR-AUC | 0.973 |
| Precision@1000 | 0.996 |
| Best F1 | 0.935 |

## API Endpoints

| Endpoint | Method | ML Function |
|---|---|---|
| `/api/health` | GET | Model loading status |
| `/api/model` | GET | Model metadata |
| `/api/pipeline` | GET | Pipeline stage status |
| `/api/evaluation` | GET | Evaluation metrics |
| `/api/neurons/{id}` | GET | `get_neuron()` |
| `/api/neurons/search` | GET | Search by name/type |
| `/api/predict` | POST | `score_connection()` |
| `/api/candidates` | POST | `rank_candidate_targets()` |
| `/api/research/summary` | GET | Research summary |

## Scientific Language

All predictions are described as:

- "Model-suggested connection score"
- "Candidate connection score"
- NOT: "biological probability", "guaranteed connection", "proof of connection"

## Memory Safety

- Model loaded once, cached in memory
- Feature construction uses chunked processing (50,000 pairs/chunk)
- Candidate ranking uses bounded pool size (max 10,000)
- No raw data loading into frontend
- No all-pairs matrix construction
