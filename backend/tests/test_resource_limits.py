"""Tests for configured resource caps and schema ceilings."""

from __future__ import annotations

from app.core.config import settings


class TestCandidatesCaps:
    def test_k_above_configured_max_is_429(self, client):
        assert settings.MAX_K == 100
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": settings.MAX_K + 1, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["code"] == "LIMIT_EXCEEDED"

    def test_pool_above_configured_max_is_429(self, client):
        assert settings.MAX_CANDIDATES == 1000
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": 10, "candidate_pool_size": settings.MAX_CANDIDATES * 2},
        )
        assert resp.status_code == 429

    def test_pool_beyond_schema_ceiling_is_422(self, client):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": 10, "candidate_pool_size": 20000},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_zero_k_is_422(self, client):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": 0, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 422

    def test_negative_pool_is_422(self, client):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": 10, "candidate_pool_size": -5},
        )
        assert resp.status_code == 422

    def test_non_positive_source_root_is_422(self, client):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 0, "k": 10, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 422


class TestPredictPayloadBounds:
    def test_negative_ids_422(self, client):
        resp = client.post("/api/predict", json={"source_root_id": -1, "target_root_id": 1})
        assert resp.status_code == 422

    def test_zero_source_422(self, client):
        resp = client.post("/api/predict", json={"source_root_id": 0, "target_root_id": 1})
        assert resp.status_code == 422

    def test_floating_ids_422(self, client):
        resp = client.post("/api/predict", json={"source_root_id": 1.5, "target_root_id": 1})
        assert resp.status_code == 422

    def test_string_ids_422(self, client):
        resp = client.post("/api/predict", json={"source_root_id": "one", "target_root_id": 1})
        assert resp.status_code == 422