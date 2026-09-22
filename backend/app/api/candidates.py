"""Candidate ranking endpoint."""

import math

from fastapi import APIRouter

from app.core.config import settings
from app.core.errors import LimitExceededError, FlyMindError, InternalError, NeuronNotFoundError
from app.schemas.models import CandidatesRequest, CandidatesResponse, CandidateItem
from app.services.ml_service import ml_service

router = APIRouter()


def _clean_str(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return str(val)


@router.post("/candidates", response_model=CandidatesResponse)
def rank_candidates(req: CandidatesRequest) -> CandidatesResponse:
    if req.k > settings.MAX_K:
        raise LimitExceededError(
            f"k exceeds the configured maximum of {settings.MAX_K}"
        )
    if req.candidate_pool_size > settings.MAX_CANDIDATES:
        raise LimitExceededError(
            f"candidate_pool_size exceeds the configured maximum of {settings.MAX_CANDIDATES}"
        )

    try:
        candidates = ml_service.rank_candidate_targets(
            req.source_root_id,
            k=req.k,
            candidate_pool_size=req.candidate_pool_size,
        )
    except KeyError:
        raise NeuronNotFoundError("Source neuron was not found")
    except FlyMindError:
        raise
    except Exception:
        raise InternalError("Candidate ranking service is not available")

    items = [
        CandidateItem(
            rank=c.rank,
            source_root_id=c.source_root_id,
            target_root_id=c.target_root_id,
            score=c.score,
            target_nt_type=_clean_str(c.target_nt_type),
            target_super_class=_clean_str(c.target_super_class),
            target_primary_type=_clean_str(c.target_primary_type),
        )
        for c in candidates
    ]

    return CandidatesResponse(
        source_root_id=req.source_root_id,
        candidates=items,
        total_sampled=req.candidate_pool_size,
    )