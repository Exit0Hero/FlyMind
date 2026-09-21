"""Candidate ranking endpoint."""

import math

from fastapi import APIRouter, HTTPException

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
    try:
        candidates = ml_service.rank_candidate_targets(
            req.source_root_id,
            k=req.k,
            candidate_pool_size=req.candidate_pool_size,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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
