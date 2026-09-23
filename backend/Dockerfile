# FlyMind backend production image.
#
# Build context is the REPO ROOT so the image can ship both the FastAPI app
# and the frozen research inference package (src/link_prediction/*):
#
#   docker build -f backend/Dockerfile -t flymind-backend:local .
#
# Strategy
#   * Multi-stage to keep the runtime layer lean and reproducible.
#   * Installs from backend/requirements.txt inside a virtualenv.
#   * Runs as an unprivileged user (no root in the container).
#   * Does NOT bake the model artifact or any dataset — those are mounted as
#     read-only volumes at runtime (see compose.yaml and docs/DEPLOYMENT.md).
#   * Layout under /flymind mirrors the host project root so
#     settings.PROJECT_ROOT and src/link_prediction/config.py resolve the same
#     data/models/results paths after the bind mounts land.

FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv $VIRTUAL_ENV

WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLYMIND_PROJECT_ROOT=/flymind \
    PORT=8001

# Non-root runtime user. Model/dataset directories stay outside the image and
# are ownership-agnostic because we mount them read-only as ":ro".
RUN groupadd --system flymind && useradd --system --gid flymind --home-dir /flymind flymind

WORKDIR /flymind

COPY --from=builder --chown=flymind:flymind /opt/venv /opt/venv
COPY --chown=flymind:flymind backend/app ./app
COPY --chown=flymind:flymind src ./src

USER flymind

# Liveness: the process is up. Readiness is a separate, strict probe on /api/ready.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=5 \
    CMD ["sh", "-c", \
         "/opt/venv/bin/python -c \"import urllib.request,sys; \
          r=urllib.request.urlopen('http://127.0.0.1:${PORT:-8001}/api/health', timeout=2); \
          sys.exit(0 if r.status==200 else 1)\""]

EXPOSE 8001

# PORT is read from the environment (compose sets it to FLYMIND_PORT).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001} --proxy-headers"]
