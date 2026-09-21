# FlyMind — Final Application Architecture

## Pipeline Overview

```
                    ┌───────────────────┐
                    │    FlyWire Data   │
                    │  139,255 neurons  │
                    │  3.73M directed   │
                    │      edges        │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │       ETL         │
                    │  ID normalization │
                    │  Missing values   │
                    │  Edge aggregation │
                    │  Parquet storage  │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Feature Engineer  │
                    │ 15 node features  │
                    │ 60D pair features │
                    │ [xA,xB,|xA-xB|,  │
                    │    xA*xB]         │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Trained ML Model  │
                    │ Random Forest     │
                    │ 100 estimators    │
                    │ Frozen artifact   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │   Evaluation      │
                    │ ROC-AUC 0.9800    │
                    │ PR-AUC 0.9739     │
                    │ Recall@10 0.814   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │    Prediction     │
                    │ Score pairs       │
                    │ 60D → score       │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Candidate Ranking │
                    │ Rank targets      │
                    │ Exclude known     │
                    │ Return top-K      │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Streamlit App     │
                    │ 7 pages           │
                    │ Interactive       │
                    └───────────────────┘
```

## Components

### Data Layer

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| Neuron table | `data/processed/neuron_table.parquet` | 9.8 MB | 139,255 neurons x 41 columns |
| Edge table | `data/processed/link_prediction/edges_aggregated.parquet` | 17 MB | 3.73M directed edges |
| Feature matrix | `data/processed/link_prediction/X_features.npy` | 8.4 MB | 139,255 x 15 float32 |
| ID mapping | `data/processed/link_prediction/id_to_idx.json` | 4 MB | root_id to matrix index |

### Model Artifact

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| Production model | `models/link_prediction_rf.pkl` | 177 MB | Frozen RF classifier |
| Model metadata | `models/link_prediction_rf.metadata.json` | 1 KB | Version, metrics, contract |

### Feature Contract

- **Node features (15):** nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg, length_nm, area_nm, size_nm, coord_x, coord_y, coord_z, flow_enc, side_x_enc
- **Pair features (60):** [source, target, |source-target|, source*target]

### Application Pages

| Page | File | Description |
|------|------|-------------|
| Overview | `app/pages/overview.py` | Project summary, key metrics |
| Pipeline Overview | `app/pages/pipeline_overview.py` | Full ML pipeline status |
| Neuron Explorer | `app/pages/neuron_explorer.py` | Search and inspect neurons |
| Connection Predictor | `app/pages/connection_predictor.py` | Evaluate source-target pairs |
| Candidate Ranking | `app/pages/candidate_ranking.py` | Rank candidate targets |
| Research Results | `app/pages/research_results.py` | Experiment results, evaluation |
| About / Limitations | `app/pages/about.py` | Scientific context, limitations |

### Data Flow

1. User enters source/target neuron IDs
2. Lookup 15-dim feature vectors from precomputed matrix
3. Build 60-dim pair features: [xA, xB, |xA-xB|, xA*xB]
4. RF model predicts connection score
5. Score displayed with interpretation disclaimer

### Memory Safety

- Feature matrix precomputed and cached (8.4 MB)
- Model loaded once and cached via `@st.cache_resource`
- Candidate ranking uses chunked inference (10K candidates per chunk)
- No full pair matrices constructed at startup
- Max candidate limit enforced (1000)

### Test Suite

- 99 tests across 6 test files
- All passing
- Covers: data integrity, pipeline, link prediction, cold start, inference, production inference
