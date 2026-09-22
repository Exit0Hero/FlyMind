"""FlyMind FastAPI backend entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import init_exception_handlers
from app.core.logging import getLogger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.api import (
    health,
    ready,
    model,
    pipeline,
    evaluation,
    neurons,
    predict,
    candidates,
    research,
)

log = getLogger("flymind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The ML model loads lazily on first request to keep cold start light.
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Exception handlers FIRST so they can be wrapped by everything else.
    init_exception_handlers(app)

    # Middleware ordering: last-added is outermost. CORS needs to be the
    # outermost layer so CORS headers are present even on error responses.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(ready.router, prefix="/api", tags=["health"])
    app.include_router(model.router, prefix="/api", tags=["model"])
    app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])
    app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])
    app.include_router(neurons.router, prefix="/api", tags=["neurons"])
    app.include_router(predict.router, prefix="/api", tags=["predict"])
    app.include_router(candidates.router, prefix="/api", tags=["candidates"])
    app.include_router(research.router, prefix="/api", tags=["research"])

    setup_logging()
    log.info(
        "service started",
        extra={
            "fields": {
                "env": settings.ENV,
                "version": settings.APP_VERSION,
                "model_version": settings.MODEL_VERSION,
                "debug": settings.DEBUG,
            }
        },
    )
    return app


app = create_app()