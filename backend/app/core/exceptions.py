"""Global exception handlers producing structured, leak-free error payloads.

Every error response uses the shape:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

Codes and statuses follow the API contract (400/404/422/429/500/503) with
stable machine-readable codes (e.g. INVALID_NEURON). Internal exceptions are
logged in full server-side but only a generic message reaches the client.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import FlyMindError
from app.core.logging import setup_logging

log = setup_logging()

_HTTP_CODE_TO_DEFAULT_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
    429: "LIMIT_EXCEEDED",
    503: "SERVICE_UNAVAILABLE",
}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _error_response(
    request: Request,
    *,
    code: str,
    message: str,
    status_code: int,
    details: list[Any] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    req_id = _request_id(request)
    if req_id:
        error["request_id"] = req_id
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


def _register_handlers(app: FastAPI) -> None:
    @app.exception_handler(FlyMindError)
    async def flymind_error_handler(request: Request, exc: FlyMindError):
        log.error(
            "api error",
            extra={
                "fields": {
                    "request_id": _request_id(request),
                    "code": exc.code,
                    "status": exc.status_code,
                }
            },
        )
        return _error_response(
            request, code=exc.code, message=exc.message, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Only expose field paths + messages, never echoed input values.
        errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            errors.append({"field": loc or "$", "message": err.get("msg", "invalid value")})
        log.warning(
            "validation error",
            extra={
                "fields": {
                    "request_id": _request_id(request),
                    "count": len(errors),
                    "status": 422,
                }
            },
        )
        return _error_response(
            request,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=422,
            details=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code = _HTTP_CODE_TO_DEFAULT_CODE.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail
        if isinstance(detail, dict):
            detail = detail.get("detail", "Request failed")
        log.warning(
            "http error",
            extra={
                "fields": {
                    "request_id": _request_id(request),
                    "status": exc.status_code,
                    "code": code,
                }
            },
        )
        return _error_response(request, code=code, message=str(detail), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.exception(
            "unhandled exception",
            extra={"fields": {"request_id": _request_id(request), "path": request.url.path}},
        )
        return _error_response(
            request,
            code="INTERNAL_ERROR",
            message="Internal server error",
            status_code=500,
        )


def init_exception_handlers(app: FastAPI) -> None:
    _register_handlers(app)