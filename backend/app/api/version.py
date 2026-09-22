"""Application / API / model version surface.

Serves the version identifiers that operators and release tooling need, without
ever touching the model artifact:

* ``app_version`` — the deployed application code (FastAPI service).
* ``api_version`` — the HTTP API surface contract.
* ``model_version`` — the **model artifact** version, read from the settings /
  artifact metadata. It is deliberately kept separate from the application
  version: deploying a new frontend or API build must never imply a new model,
  and a new model artifact must never bump the application version.
* ``model_loaded`` — whether the artifact is currently loaded into memory (so a
  client can tell "version known but service not ready" apart).

This endpoint does not load the model and never blocks on ML warm-up, so it is
safe to call from CI smoke tests and from pre-rollout checks.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.models import VersionResponse
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/version", response_model=VersionResponse, summary="Version identifiers")
def version() -> VersionResponse:
    return VersionResponse(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        api_version=settings.APP_VERSION,  # kept in sync with the API contract
        model_version=ml_service.model_version,
        model_loaded=ml_service.is_loaded,
    )
