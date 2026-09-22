"""Model info endpoint."""

from fastapi import APIRouter

from app.core.errors import FlyMindError, InternalError
from app.schemas.models import ModelInfoResponse
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/model", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    try:
        info = ml_service.model_info
        return ModelInfoResponse(
            model_type="RandomForestClassifier",
            model_version=ml_service.model_version,
            n_estimators=info["n_estimators"],
            feature_dim=ml_service.feature_dim,
            node_feature_dim=ml_service.node_feature_dim,
            n_features=len(ml_service.feature_names),
            feature_names=ml_service.feature_names,
            n_classes=info["n_classes"],
            class_labels=info["class_labels"],
            n_neurons=ml_service.n_neurons,
            n_edges=ml_service.n_edges,
        )
    except FlyMindError:
        raise
    except Exception:
        raise InternalError("Model service is not available")