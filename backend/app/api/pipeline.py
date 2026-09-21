"""Pipeline status endpoint."""

from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.models import PipelineResponse, PipelineStage
from app.services.ml_service import ml_service

router = APIRouter()


def _check_model_file() -> PipelineStage:
    p = settings.MODELS_DIR / "link_prediction_rf.pkl"
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        return PipelineStage(name="trained_model", status="ready", detail=f"{size_mb:.1f} MB")
    return PipelineStage(name="trained_model", status="missing", detail=str(p))


def _check_features() -> PipelineStage:
    lp = settings.LP_DIR
    required = ["X_features.npy", "id_to_idx.json", "edges_aggregated.parquet"]
    missing = [f for f in required if not (lp / f).exists()]
    if missing:
        return PipelineStage(name="processed_features", status="missing", detail=f"Missing: {', '.join(missing)}")
    return PipelineStage(name="processed_features", status="ready")


def _check_neuron_table() -> PipelineStage:
    p = settings.PROCESSED_DIR / "neuron_table.parquet"
    if p.exists():
        return PipelineStage(name="neuron_table", status="ready")
    return PipelineStage(name="neuron_table", status="missing")


def _check_reports() -> PipelineStage:
    reports_dir = settings.REPORTS_DIR
    if not reports_dir.exists():
        return PipelineStage(name="evaluation_reports", status="missing")
    json_files = list(reports_dir.glob("*.json"))
    if json_files:
        return PipelineStage(name="evaluation_reports", status="ready", detail=f"{len(json_files)} reports")
    return PipelineStage(name="evaluation_reports", status="missing")


def _check_ml_loading() -> PipelineStage:
    if ml_service.is_loaded:
        return PipelineStage(name="ml_inference", status="ready", detail=f"{ml_service.n_neurons} neurons")
    if ml_service.load_error:
        return PipelineStage(name="ml_inference", status="error", detail=ml_service.load_error)
    return PipelineStage(name="ml_inference", status="missing", detail="Not yet loaded")


@router.get("/pipeline", response_model=PipelineResponse)
def get_pipeline_status() -> PipelineResponse:
    stages = [
        _check_model_file(),
        _check_features(),
        _check_neuron_table(),
        _check_reports(),
        _check_ml_loading(),
    ]
    return PipelineResponse(stages=stages)
