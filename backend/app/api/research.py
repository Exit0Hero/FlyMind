"""Research summary endpoint."""

from fastapi import APIRouter

from app.schemas.models import ResearchSummaryResponse
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/research/summary", response_model=ResearchSummaryResponse)
def get_research_summary() -> ResearchSummaryResponse:
    n_neurons = ml_service.n_neurons if ml_service.is_loaded else 0
    n_edges = ml_service.n_edges if ml_service.is_loaded else 0

    return ResearchSummaryResponse(
        title="FlyMind: Computational Link Prediction in Drosophila Connectome",
        description=(
            "FlyMind is a machine learning pipeline for predicting synaptic "
            "connections in the Drosophila melanogaster whole-brain connectome. "
            "It uses a Random Forest classifier trained on node-level features "
            "(morphology, neurotransmitter type, spatial coordinates) to rank "
            "candidate directed connections."
        ),
        dataset={
            "n_neurons": n_neurons,
            "n_edges": n_edges,
            "organism": "Drosophila melanogaster",
            "data_source": "FlyWire whole-brain connectome",
        },
        experiments={
            "2a": "Link prediction (train/val/test split)",
            "2b": "Cold-start generalization",
            "2d": "Feature importance analysis",
            "3": "Robustness analysis (per-source ranking, calibration, held-out ranks)",
        },
        key_findings=[
            "RF model with node features achieves strong link prediction performance",
            "Cold-start split validates generalization to unseen neurons",
            "Neurotransmitter type and morphology features are most important",
            "Model produces well-calibrated probability estimates",
            "Per-source ranking shows consistent performance across neurons",
        ],
    )
