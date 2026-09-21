"""FlyMind FastAPI backend entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import health, model, pipeline, evaluation, neurons, predict, candidates, research


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-import to trigger lazy loading on startup if desired
    # The ML model loads lazily on first request; comment out to preload:
    # from app.services.ml_service import ml_service; ml_service._load()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(model.router, prefix="/api", tags=["model"])
app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])
app.include_router(neurons.router, prefix="/api", tags=["neurons"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(candidates.router, prefix="/api", tags=["candidates"])
app.include_router(research.router, prefix="/api", tags=["research"])
