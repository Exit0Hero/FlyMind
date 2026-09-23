"""Readiness probe.

Returns 200 only when the service can serve model-backed requests. Anything
less (model missing, integrity failure, not yet loaded) returns 503 with a
client-safe reason.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.core.errors import NotReadyError
from app.core.logging import getLogger
from app.services.ml_service import ml_service

router = APIRouter()
log = getLogger("flymind.ready")


@router.get("/ready", summary="Readiness probe")
def readiness() -> dict[str, str]:
    # Warm-up: readiness is the first (and only) probe allowed to trigger a
    # model load. /api/health and /api/version stay metadata-only so CI and
    # preflight checks never pay the 177 MB cold-start cost by accident.
    ml_service.ensure_load()

    reasons: list[str] = []
    if not ml_service.is_loaded:
        reasons.append(ml_service.load_error or "model not loaded")
        model_path = settings.MODEL_PATH
        if not model_path.exists():
            reasons.append(f"model artifact {model_path.name} missing")

    if reasons:
        log.info("service not ready", extra={"fields": {"reasons": reasons}})
        raise NotReadyError(" ".join(reasons))

    return {"status": "ready"}