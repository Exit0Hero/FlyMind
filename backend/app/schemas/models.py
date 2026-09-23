"""Pydantic request/response models for FlyMind API."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer


def _coerce_root_id(value: Any) -> int:
    """Accept int or digit-string root IDs and normalize to int.

    FlyWire root IDs exceed JS Number.MAX_SAFE_INTEGER, so the browser must
    receive them as JSON strings; requests may send either form.
    """
    if isinstance(value, bool):
        raise ValueError("root id must be an integer")  # noqa: TRY004 — pydantic BeforeValidator contract
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ValueError("root id must be an integer")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("root id must be an integer") from exc
    raise ValueError("root id must be an integer")


# JSON-safe root ID: int internally, string on the wire (JS precision).
RootId = Annotated[
    int,
    BeforeValidator(_coerce_root_id),
    PlainSerializer(lambda v: str(v), return_type=str),
]


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    n_neurons: int | None = None
    n_edges: int | None = None
    version: str = ""


class VersionResponse(BaseModel):
    """Versioning surface: application, API, and model versions are distinct.

    ``model_version`` is read from the artifact **metadata** (never the loaded
    model), so a client can identify the served artifact without triggering a
    model load. Application and API versions describe the running code; they
    do NOT change when a new model artifact is deployed, and vice-versa.
    """

    app_name: str
    app_version: str
    api_version: str
    model_version: str | None = None
    model_artifact: str | None = None
    model_loaded: bool = False


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    model_type: str = "RandomForestClassifier"
    model_version: str | None = None
    n_estimators: int | None = None
    feature_dim: int | None = None
    node_feature_dim: int | None = None
    n_features: int | None = None
    feature_names: list[str] = []
    n_classes: int | None = None
    class_labels: list[int] = []
    n_neurons: int | None = None
    n_edges: int | None = None


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------
class PipelineStage(BaseModel):
    name: str
    status: str  # "ready" | "missing" | "error"
    detail: str = ""


class PipelineResponse(BaseModel):
    stages: list[PipelineStage]


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------
class ExperimentResult(BaseModel):
    experiment: str
    data: dict[str, Any]


class EvaluationResponse(BaseModel):
    experiments: list[ExperimentResult]


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------
class NeuronResponse(BaseModel):
    root_id: RootId
    nt_type: str | None = None
    nt_type_score: float | None = None
    primary_type: str | None = None
    super_class: str | None = None
    flow: str | None = None
    coord_x: float | None = None
    coord_y: float | None = None
    coord_z: float | None = None
    length_nm: float | None = None
    area_nm: float | None = None
    size_nm: float | None = None
    side: str | None = None
    name: str | None = None
    outgoing_count: int | None = None
    incoming_count: int | None = None


class NeuronSearchItem(BaseModel):
    root_id: RootId
    name: str | None = None
    nt_type: str | None = None
    super_class: str | None = None
    primary_type: str | None = None


class NeuronSearchResponse(BaseModel):
    query: str
    results: list[NeuronSearchItem]
    total: int


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    source_root_id: RootId = Field(..., gt=0, description="Presynaptic neuron root ID")
    target_root_id: RootId = Field(..., gt=0, description="Postsynaptic neuron root ID")


class PredictResponse(BaseModel):
    source_root_id: RootId
    target_root_id: RootId
    score: float
    is_known_edge: bool
    is_self_loop: bool


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
class CandidatesRequest(BaseModel):
    source_root_id: RootId = Field(..., gt=0, description="Presynaptic neuron root ID")
    # Hard memory ceilings; the configured FLYMIND_MAX_K / FLYMIND_MAX_CANDIDATES
    # operational caps are enforced in the endpoint (429).
    k: int = Field(10, ge=1, le=1000, description="Number of top candidates to return")
    candidate_pool_size: int = Field(1000, ge=100, le=10000)


class CandidateItem(BaseModel):
    rank: int
    source_root_id: RootId
    target_root_id: RootId
    score: float
    target_nt_type: str | None = None
    target_super_class: str | None = None
    target_primary_type: str | None = None


class CandidatesResponse(BaseModel):
    source_root_id: RootId
    candidates: list[CandidateItem]
    total_sampled: int


# ---------------------------------------------------------------------------
# Research summary
# ---------------------------------------------------------------------------
class ResearchSummaryResponse(BaseModel):
    title: str = "FlyMind: Computational Link Prediction in Drosophila Connectome"
    description: str = ""
    dataset: dict[str, Any] = {}
    experiments: dict[str, Any] = {}
    key_findings: list[str] = []
