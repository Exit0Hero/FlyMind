"""Predict endpoint for scoring directed neuron pairs."""

from fastapi import APIRouter, HTTPException

from app.schemas.models import PredictRequest, PredictResponse
from app.services.ml_service import ml_service

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_connection(req: PredictRequest) -> PredictResponse:
    try:
        result = ml_service.score_connection(req.source_root_id, req.target_root_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(
        source_root_id=result.source_root_id,
        target_root_id=result.target_root_id,
        score=result.score,
        is_known_edge=result.is_known_edge,
        is_self_loop=result.is_self_loop,
    )
