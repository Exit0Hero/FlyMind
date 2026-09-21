# Architecture Migration Audit

## Overview

FlyMind migrated from Streamlit to a full-stack architecture with:
- **Frontend**: Next.js 15 + React 19 + TypeScript + Tailwind CSS 4
- **Backend**: FastAPI (Python) wrapping existing ML inference
- **ML Layer**: Preserved exactly — no retraining, no modification

## Component Mapping

| Existing Component | Location | Status | New Location |
|---|---|---|---|
| ML model (RF) | `models/link_prediction_rf.pkl` | PRESERVED | `backend/app/services/ml_service.py` loads it |
| Inference engine | `src/link_prediction/inference.py` | PRESERVED | Called by `backend/app/services/ml_service.py` |
| Feature engineering | `src/link_prediction/features.py` | PRESERVED | Used by inference engine |
| Production inference | `src/link_prediction/inference.py:FlyMindInference` | PRESERVED | Wrapped by `MLService` |
| Candidate ranking | `src/link_prediction/inference.py:rank_candidate_targets` | PRESERVED | Exposed via `POST /api/candidates` |
| Neuron lookup | `src/link_prediction/inference.py:get_neuron` | PRESERVED | Exposed via `GET /api/neurons/{id}` |
| Streamlit UI | `app/` | ARCHIVED | `frontend/` (Next.js) |
| Streamlit app | `app/streamlit_app.py` | ARCHIVED | `frontend/app/page.tsx` |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check, model status |
| `/api/model` | GET | Model metadata, feature info |
| `/api/pipeline` | GET | Pipeline stage statuses |
| `/api/evaluation` | GET | Experiment results from reports |
| `/api/neurons/{id}` | GET | Neuron lookup by root ID |
| `/api/neurons/search?q=` | GET | Search neurons by name/ID |
| `/api/predict` | POST | Score a directed neuron pair |
| `/api/candidates` | POST | Rank candidate targets |
| `/api/research/summary` | GET | Research summary data |

## Frontend Pages

| Page | Route | Description |
|---|---|---|
| Overview | `/` | Dashboard with metrics and pipeline strip |
| ML Pipeline | `/pipeline` | Interactive pipeline visualization |
| Neuron Explorer | `/neurons` | Search and inspect neurons |
| Connection Predictor | `/predictor` | Score neuron pairs |
| Candidate Ranking | `/candidates` | Rank candidate targets |
| Research Results | `/research` | Evaluation metrics and findings |
| About | `/about` | Methodology and limitations |

## Design System

- Dark-first theme: `#0B0E14` base, `#12161F` surface
- Accent: `#4FD1C5` (teal-cyan) for observed/system
- Violet: `#8B7CF6` for predicted/candidate
- Fonts: Inter (UI), Inter Tight (display), JetBrains Mono (code)
- Border radius: 6px/10px/16px

## ML Preservation

- Model unchanged: `link_prediction_rf.pkl` (177MB RF)
- Feature contract: 15 node features → 60D pair features
- Inference: `FlyMindInference.score_connection()` and `rank_candidate_targets()`
- All 45 existing tests pass
- No retraining performed
- No data modification

## Running Services

```bash
# Backend (port 8001)
cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# Frontend (port 3000)
cd frontend && npm run dev
```
