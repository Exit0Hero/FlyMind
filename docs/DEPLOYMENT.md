# FlyMind Deployment Guide

## Architecture

```
Browser
   ↓
Streamlit (port 8501)
   ↓
Production Inference Layer
   ↓
Frozen RF Model (177 MB)
   ↓
Processed FlyMind Data (~40 MB)
```

Self-contained Streamlit application. No external API service.

## Prerequisites

- Docker 20.10+ (for containerized deployment)
- Python 3.11+ (for local development)
- 512 MB RAM minimum

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLYMIND_DATA_ROOT` | `./data/processed` | Processed data directory |
| `FLYMIND_MODEL_DIR` | `./models` | Model artifact directory |
| `FLYMIND_MODEL_VERSION` | `FlyMind-RF v1.0.0` | Model version string |
| `FLYMIND_MAX_CANDIDATES` | `1000` | Max candidate ranking count |
| `FLYMIND_LOG_LEVEL` | `INFO` | Logging level |

## Docker Build

```bash
docker build -t flymind:latest .
```

## Local Run

```bash
docker run -p 8501:8501 flymind:latest
```

Dashboard opens at http://localhost:8501

## Health Validation

```bash
docker exec <container_id> python -c "
from src.inference.health_check import check_model_health
from pathlib import Path
h = check_model_health(Path('models'), Path('data/processed/link_prediction'))
print(f'Status: {h.status}')
print(f'Details: {h.details}')
"
```

## Local Development

```bash
pip install -r requirements.txt
pip install streamlit
streamlit run app/streamlit_app.py
```

## Smoke Testing

1. Overview page loads with metrics
2. Neuron Explorer: search neuron `720575940597856265`
3. Connection Predictor: source `720575940597856265`, target `720575940602380768`
4. Candidate Ranking: source `720575940597856265`, count 10
5. Research Results: all metrics visible
6. About: limitations displayed

## Logs

```bash
docker logs <container_id>
```

## Troubleshooting

**Container won't start:**
- Check Docker logs: `docker logs <container_id>`
- Verify model artifact exists: `ls -la models/link_prediction_rf.pkl`
- Check memory: ensure 512 MB available

**Dashboard shows errors:**
- Run health check (see above)
- Verify processed data exists in container

**Port conflict:**
- Use different port: `docker run -p 8502:8501 flymind:latest`

## Rollback

See `docs/ROLLBACK.md`
