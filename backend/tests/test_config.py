"""Tests for application configuration and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import DEFAULT_CORS_DEV, Settings


def _make(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_defaults():
    s = _make()
    assert s.ENV == "dev"
    assert s.APP_NAME == "FlyMind API"
    assert s.MAX_CANDIDATES == 1000
    assert s.MAX_K == 100
    assert s.CORS_ORIGINS == DEFAULT_CORS_DEV
    assert s.MODEL_VERSION == "FlyMind-RF v1.0.0"


def test_cors_json_string():
    s = _make(CORS_ORIGINS='["http://a.example","http://b.example"]')
    assert s.CORS_ORIGINS == ["http://a.example", "http://b.example"]


def test_cors_comma_string():
    s = _make(CORS_ORIGINS="http://a.example, http://b.example")
    assert s.CORS_ORIGINS == ["http://a.example", "http://b.example"]


def test_cors_invalid_json_raises():
    with pytest.raises(ValidationError):
        _make(CORS_ORIGINS='{"not": "array"}')


def test_prod_forbids_wildcard_cors():
    with pytest.raises(ValidationError):
        _make(ENV="prod", CORS_ORIGINS=["*"])


def test_dev_allows_localhost_only():
    s = _make(CORS_ORIGINS=["http://localhost:3000"])
    assert s.CORS_ORIGINS == ["http://localhost:3000"]


def test_max_k_must_fit_within_max_candidates():
    with pytest.raises(ValidationError):
        _make(MAX_K=2000, MAX_CANDIDATES=100)


def test_invalid_log_level_raises():
    with pytest.raises(ValidationError):
        _make(LOG_LEVEL="VERBOSE")


def test_invalid_env_raises():
    with pytest.raises(ValidationError):
        _make(ENV="staging")


def test_port_bounds():
    with pytest.raises(ValidationError):
        _make(PORT=70000)


def test_model_path_derived():
    s = _make(MODELS_DIR="models")
    assert s.MODEL_PATH.name == "link_prediction_rf.pkl"
    assert s.MODEL_METADATA_PATH.name == "link_prediction_rf.metadata.json"