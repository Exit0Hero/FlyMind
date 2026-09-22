"""Structured API error types.

Errors raised in the application layer carry a stable machine-readable
``code`` and an HTTP ``status_code``. The global exception handlers in
``app.core.exceptions`` translate them into a consistent response body:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

Messages must be safe to expose to a client: no stack traces, no absolute
filesystem paths, no internal variable dumps.
"""

from __future__ import annotations

from typing import Any


class FlyMindError(Exception):
    """Base class for all intended FlyMind API errors."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code

    @property
    def message(self) -> str:
        return str(self)

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if request_id:
            payload["error"]["request_id"] = request_id
        return payload


class ValidationFailedError(FlyMindError):
    """Request payload failed one of the configured business rules."""

    status_code = 422
    code = "VALIDATION_ERROR"


class LimitExceededError(FlyMindError):
    """Request exceeds a configured resource cap (429)."""

    status_code = 429
    code = "LIMIT_EXCEEDED"


class NeuronNotFoundError(FlyMindError):
    """The referenced neuron does not exist in the dataset."""

    status_code = 404
    code = "INVALID_NEURON"


class NotFoundError(FlyMindError):
    """A requested resource is not available."""

    status_code = 404
    code = "NOT_FOUND"


class ModelUnavailableError(FlyMindError):
    """The ML model could not be loaded or is not ready (503)."""

    status_code = 503
    code = "MODEL_UNAVAILABLE"


class NotReadyError(FlyMindError):
    """The service is not ready to serve traffic (503)."""

    status_code = 503
    code = "NOT_READY"


class InternalError(FlyMindError):
    """An unexpected internal error occurred (500). Message is generic."""

    status_code = 500
    code = "INTERNAL_ERROR"