"""Application configuration via environment variables.

All settings read from the environment using the ``FLYMIND_`` prefix. Any
compile-time constraint that can fail (bad enum, path that resolves to
nowhere, CORS contract violations, invalid limits) fails at import time with a
clear message, so a misconfigured instance never serves ambiguous behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

AppEnv = Literal["dev", "test", "prod"]
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
DEFAULT_CORS_DEV = ["http://localhost:3000", "http://127.0.0.1:3000"]


class Settings(BaseSettings):
    # --- identity ----------------------------------------------------------
    ENV: AppEnv = "dev"
    APP_NAME: str = "FlyMind API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = Field(8000, ge=1, le=65535)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

    # --- CORS ---------------------------------------------------------------
    # A JSON array string or a comma-separated list; SEE .env.example.
    CORS_ORIGINS: list[str] = DEFAULT_CORS_DEV

    # --- paths --------------------------------------------------------------
    DATA_ROOT: Path | None = Field(default=None)
    MODELS_DIR: Path | None = Field(default=None)
    REPORTS_DIR: Path | None = Field(default=None)

    # --- model artifacts ----------------------------------------------------
    MODEL_FILENAME: str = "link_prediction_rf.pkl"
    MODEL_METADATA_FILENAME: str = "link_prediction_rf.metadata.json"
    MODEL_VERSION: str = "FlyMind-RF v1.0.0"
    ENABLE_MODEL_INTEGRITY_CHECK: bool = True

    # --- resource limits ----------------------------------------------------
    MAX_CANDIDATES: int = Field(1000, ge=100, le=10000)  # candidate_pool_size cap
    MAX_K: int = Field(100, ge=1, le=100)  # top-k cap
    DEFAULT_CANDIDATES: int = Field(100, ge=1, le=10000)
    MAX_SEARCH_LIMIT: int = Field(100, ge=1, le=500)
    MAX_SEARCH_LENGTH: int = Field(128, ge=1, le=2000)

    # --- observability ------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_prefix": "FLYMIND_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _coerce_cors(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError("FLYMIND_CORS_ORIGINS must be a JSON array string") from exc
                if not isinstance(parsed, list) or not all(isinstance(o, str) for o in parsed):
                    raise ValueError("FLYMIND_CORS_ORIGINS must be a JSON array of strings")
                return parsed
            if value.startswith("{"):
                raise ValueError("FLYMIND_CORS_ORIGINS must be a JSON array, not an object")
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value):
        value = value.upper()
        if value not in _LOG_LEVELS:
            raise ValueError(f"FLYMIND_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")
        return value

    @property
    def SRC_DIR(self) -> Path:
        return self.PROJECT_ROOT / "src"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed"

    @property
    def LP_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed" / "link_prediction"

    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / self.MODEL_FILENAME

    @property
    def MODEL_METADATA_PATH(self) -> Path:
        return self.MODELS_DIR / self.MODEL_METADATA_FILENAME

    def model_post_init(self, __context, /) -> None:
        if self.DATA_ROOT is None or not self.DATA_ROOT.exists():
            object.__setattr__(self, "DATA_ROOT", self.PROJECT_ROOT / "data")
        if self.MODELS_DIR is None or not self.MODELS_DIR.exists():
            object.__setattr__(self, "MODELS_DIR", self.PROJECT_ROOT / "models")
        if self.REPORTS_DIR is None or not self.REPORTS_DIR.exists():
            object.__setattr__(self, "REPORTS_DIR", self.PROJECT_ROOT / "results" / "reports")

        if self.MAX_K < 1:
            raise ValueError("FLYMIND_MAX_K must be >= 1")
        if self.MAX_K > self.MAX_CANDIDATES:
            raise ValueError("FLYMIND_MAX_K must be <= FLYMIND_MAX_CANDIDATES")
        if self.ENV == "prod" and "*" in self.CORS_ORIGINS:
            raise ValueError("FLYMIND_CORS_ORIGINS wildcard '*' is not allowed in production")


settings = Settings()
