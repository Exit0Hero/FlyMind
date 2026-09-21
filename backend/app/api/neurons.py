"""Neuron lookup and search endpoints."""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.models import (
    NeuronResponse,
    NeuronSearchItem,
    NeuronSearchResponse,
)
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/neurons/search", response_model=NeuronSearchResponse)
def search_neurons(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    try:
        results = ml_service.search_neurons(q, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    items = [
        NeuronSearchItem(
            root_id=r["root_id"],
            name=r.get("name"),
            nt_type=r.get("nt_type"),
            super_class=r.get("super_class"),
        )
        for r in results
    ]
    return NeuronSearchResponse(query=q, results=items, total=len(items))


@router.get("/neurons/{root_id}", response_model=NeuronResponse)
def get_neuron(root_id: int) -> NeuronResponse:
    try:
        neuron = ml_service.get_neuron(root_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Neuron {root_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return NeuronResponse(
        root_id=neuron.root_id,
        nt_type=neuron.nt_type,
        nt_type_score=neuron.nt_type_score,
        primary_type=neuron.primary_type,
        super_class=neuron.super_class,
        flow=neuron.flow,
        coord_x=neuron.coord_x,
        coord_y=neuron.coord_y,
        coord_z=neuron.coord_z,
        length_nm=neuron.length_nm,
        area_nm=neuron.area_nm,
        size_nm=neuron.size_nm,
        side=neuron.side,
        name=neuron.name,
    )
