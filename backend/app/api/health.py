"""Health check endpoint.

Reports the liveness/load state of the service without ever triggering a
model load. Use /api/ready for readiness (model-backed traffic).
"""

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import getLogger
from app.schemas.models import HealthResponse
from app.services.ml_service import ml_service

log = getLogger("flymind.health")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    n_neurons = None
    n_edges = None
    if ml_service.is_loaded:
        try:
            n_neurons = ml_service.n_neurons
            n_edges = ml_service.n_edges
        except Exception:
            log.warning("failed to read neuron/edge counts", exc_info=True)

    return HealthResponse(
        status="ok" if ml_service.is_loaded else "degraded",
        model_loaded=ml_service.is_loaded,
        n_neurons=n_neurons,
        n_edges=n_edges,
        version=settings.APP_VERSION,
    )