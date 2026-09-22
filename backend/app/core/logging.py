"""Structured application logging.

Log lines are JSON-serializable key=value records with a stable layout so
they can be shipped to log aggregation without an additional parser.

Rules:
- never log credentials, bearer tokens, or full request bodies;
- the optional ``request_id`` field ties a record to a specific request.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Mapping, Optional

from app.core.config import settings

_LOGGER_NAME = "flymind"


class _StructuredFormatter(logging.Formatter):
    """Render records as single-line key=value JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping) and extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure the FlyMind root logger and return it.

    Safe to call more than once; reconfiguration only happens if the root
    logger has no handlers yet.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel((level or settings.LOG_LEVEL or "INFO").upper())
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def getLogger(name: str = _LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)