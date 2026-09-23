"""Request-scoped middleware: request IDs, access logging, security headers."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import getLogger

log = getLogger("flymind.middleware")


def _request_id(request: Request) -> str:
    """Use the caller-provided ID if present (client correlation), else mint one."""
    incoming = request.headers.get("X-Request-ID")
    if incoming and len(incoming) <= 128:
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, secure response headers, and an access log record.

    The request ID is available on ``request.state.request_id`` throughout
    the request life cycle, including exception handlers.
    """

    async def dispatch(self, request: Request, call_next):
        request.state.request_id = _request_id(request)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0

        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        try:
            del response.headers["Server"]
        except (KeyError, TypeError):
            pass

        fields = {
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status": getattr(response, "status_code", 0),
            "duration_ms": round(duration_ms, 2),
        }
        log.debug("http request", extra={"fields": fields})
        return response