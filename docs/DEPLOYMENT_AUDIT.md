# FlyMind Deployment Audit

**Date:** 2026-09-20
**Branch:** `feature/flymind-deployment`

## Current Architecture

| Component | Status |
|-----------|--------|
| Application | Streamlit dashboard |
| Inference | Phase 9 production-hardened service |
| Model | Frozen RF (177 MB) |
| Data | Processed FlyMind data |
| Container | Not yet created |
| CI/CD | Not yet created |

## Runtime Dependencies

### Required at runtime

| File | Size | Purpose |
|------|------|---------|
| `models/link_prediction_rf.pkl` | 177 MB | Frozen RF model |
| `models/link_prediction_rf.metadata.json` | 1 KB | Model metadata |
| `data/processed/neuron_table.parquet` | 9.4 MB | Neuron metadata |
| `data/processed/link_prediction/X_features.npy` | 8 MB | Node features |
| `data/processed/link_prediction/id_to_idx.json` | 3.9 MB | ID mapping |
| `data/processed/link_prediction/edges_aggregated.parquet` | 17 MB | Edge index |
| `results/reports/*.json` | ~30 KB | Experiment results |
| `results/figures/**/*.png` | ~1 MB | Dashboard figures |
| `results/presentation/*.png` | ~460 KB | Presentation figures |
| `src/` | ~872 KB | Python source |
| `app/` | ~168 KB | Dashboard code |
| **Total runtime** | **~217 MB** | |

### Not required at runtime

| File | Size | Reason |
|------|------|--------|
| `data/processed/edges_buhmann.parquet` | 100 MB | Research-only |
| `data/processed/edges_princeton.parquet` | 34 MB | Research-only |
| `data/processed/link_prediction/cold_start/` | 163 MB | Evaluation data |
| `data/processed/link_prediction/*.npy` (training splits) | ~250 MB | Training data |
| `data/raw/` | ~0 MB | Raw dataset (external) |
| `notebooks/` | ~0 MB | Development only |
| `tests/` | ~508 KB | Development only |
| `docs/` | ~50 KB | Development only |

## Repository Size

| Category | Size |
|----------|------|
| Model artifact | 177 MB |
| Processed data | 664 MB |
| Source code | ~1.5 MB |
| Results | ~6.4 MB |
| **Total tracked** | **~804 MB** |

## Deployment Blockers

| Blocker | Severity | Resolution |
|---------|----------|------------|
| No Dockerfile | High | Created |
| No .dockerignore | Medium | Created |
| streamlit missing from requirements.txt | High | Fixed |
| No .streamlit config | Medium | Created |
| No CI/CD | Medium | Created |

## Recommended Architecture

**Option A: Streamlit Container**

Self-contained Docker image with:
- Python 3.11 slim base
- All runtime data baked in
- Non-root user
- Health check
- Streamlit headless mode

Expected container size: ~400 MB (with dependencies)
Expected RAM: ~300 MB
Startup time: ~10-15 seconds
