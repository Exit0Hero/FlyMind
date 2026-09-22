# FlyMind backend production image.
#
# Strategy
#   * Multi-stage to keep the runtime layer lean and reproducible.
#   * Installs from a pinned requirements.txt inside a virtualenv so the image
#     is isolated from the host toolchain and works for any base image.
#   * Runs as an unprivileged user (no root in the container).
#   * Does NOT bake the model artifact or any dataset — those are mounted as
#     read-only volumes at runtime (see compose.yaml and docs/DEPLOYMENT.md).
#   * Healthcheck targets /api/health liveness and /api/ready readiness;
#     readiness explicitly signals the model+artifact state to orchestrators.

FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv $VIRTUAL_ENV

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root runtime user. Model/dataset directories stay outside the image and
# are ownership-agnostic because we mount them read-only as ":ro".
RUN groupadd --system flymind && useradd --system --gid flymind --home-dir /app flymind

WORKDIR /app

COPY --from=builder --chown=flymind:flymind /opt/venv /opt/venv
COPY --chown=flymind:flymind app ./app

USER flymind

# Liveness: the process is up. Readiness is a separate, strict probe.
# Uses ${PORT} so the probe matches the uvicorn listener whether it is the
# image default (8000) or an operator-set value (compose overrides to 8001).
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD ["sh", "-c", \
         "/opt/venv/bin/python -c \"import urllib.request,sys; \
          r=urllib.request.urlopen('http://127.0.0.1:${PORT:-8000}/api/health', timeout=2); \
          sys.exit(0 if r.status==200 else 1)\""]

EXPOSE 8000

# PORT is configurable via env (Settings.PORT); uvicorn honours the setting via
# the config module default, so we pass the env value through here.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
