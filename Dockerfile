# FlyMind Production Dashboard
# Multi-stage build for minimal image

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for pyarrow/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (production only, no torch)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# Stage 2: Production image
FROM python:3.11-slim

# Non-root user
RUN groupadd -r flymind && useradd -r -g flymind -d /app flymind

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY app/ app/

# Copy required runtime data
COPY models/link_prediction_rf.pkl models/link_prediction_rf.pkl
COPY models/link_prediction_rf.metadata.json models/link_prediction_rf.metadata.json
COPY data/processed/neuron_table.parquet data/processed/neuron_table.parquet
COPY data/processed/link_prediction/X_features.npy data/processed/link_prediction/X_features.npy
COPY data/processed/link_prediction/id_to_idx.json data/processed/link_prediction/id_to_idx.json
COPY data/processed/link_prediction/edges_aggregated.parquet data/processed/link_prediction/edges_aggregated.parquet

# Copy results for dashboard display
COPY results/reports/ results/reports/
COPY results/figures/ results/figures/
COPY results/presentation/ results/presentation/

# Set ownership
RUN chown -R flymind:flymind /app

# Switch to non-root user
USER flymind

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.inference.health_check import check_model_health; from pathlib import Path; h=check_model_health(Path('models'), Path('data/processed/link_prediction')); exit(0 if h.status=='healthy' else 1)"

# Start Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
