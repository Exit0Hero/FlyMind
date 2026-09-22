"""Evaluation results endpoint."""

import json

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.models import EvaluationResponse, ExperimentResult

router = APIRouter()

REPORT_MAP = {
    "link_prediction": "link_prediction_results.json",
    "cold_start": "link_prediction_cold_start_results.json",
    "experiment_2d": "experiment_2d_results.json",
    "experiment_3": "experiment_3_robustness.json",
    "gnn": "gnn_results.json",
    "gnn_diagnostics": "gnn_diagnostics.json",
    "graph_stats": "graph_stats.json",
    "baseline": "baseline_results.json",
}


def _load_report(filename: str) -> dict | None:
    path = settings.REPORTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@router.get("/evaluation", response_model=EvaluationResponse)
def get_evaluation(experiment: str | None = None) -> EvaluationResponse:
    experiments: list[ExperimentResult] = []

    if experiment:
        filename = REPORT_MAP.get(experiment)
        if not filename:
            raise HTTPException(status_code=400, detail=f"Unknown experiment: {experiment}. Available: {list(REPORT_MAP.keys())}")
        data = _load_report(filename)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No results found for experiment '{experiment}'")
        experiments.append(ExperimentResult(experiment=experiment, data=data))
    else:
        for key, fname in REPORT_MAP.items():
            data = _load_report(fname)
            if data is not None:
                experiments.append(ExperimentResult(experiment=key, data=data))

    return EvaluationResponse(experiments=experiments)
