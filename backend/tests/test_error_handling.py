"""Tests for structured, leak-free error handling and security headers."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Structured error contract
# ---------------------------------------------------------------------------
class TestResponseShape:
    def test_error_body_has_code_message(self, client, use_unavailable):
        resp = client.get("/api/model")
        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"
        assert body["error"]["message"]
        assert "request_id" in body["error"]

    def test_validation_error_is_422(self, client):
        resp = client.post("/api/predict", json={"source_root_id": -1, "target_root_id": 1})
        assert resp.status_code == 422
        body = resp.json()["error"]
        assert body["code"] == "VALIDATION_ERROR"
        assert any(
            err["field"] == "source_root_id" for err in body.get("details", [])
        )

    def test_malformed_payload_is_422(self, client):
        resp = client.post("/api/predict", json={"nope": 1})
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_unknown_neuron_is_404(self, client, use_stub):
        from app.services.ml_service import ml_service
        ml_service._inference.behavior["missing"] = [999]
        resp = client.get("/api/neurons/999")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "INVALID_NEURON"

    def test_predict_missing_neuron_is_404(self, client, use_stub):
        from app.services.ml_service import ml_service
        ml_service._inference.behavior["missing"] = [999]
        resp = client.post(
            "/api/predict", json={"source_root_id": 999, "target_root_id": 1}
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "INVALID_NEURON"

    def test_unknown_route_is_404(self, client):
        resp = client.get("/api/does-not-exist")
        assert resp.status_code == 404

    def test_resource_cap_is_429(self, client):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 1, "k": 500, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["code"] == "LIMIT_EXCEEDED"


class TestNoLeaks:
    def test_500_does_not_leak_internals(self, client, use_stub):
        from app.services.ml_service import ml_service
        ml_service._inference.behavior["exception"] = [7]
        resp = client.post("/api/predict", json={"source_root_id": 7, "target_root_id": 8})
        assert resp.status_code == 500
        body = resp.json()["error"]
        assert body["code"] == "INTERNAL_ERROR"
        assert "boom" not in body["message"]
        assert "tmp" not in body["message"]
        assert "/" not in body["message"]

    def test_model_not_loaded_is_503_not_500(self, client, use_unavailable):
        resp = client.get("/api/model")
        assert resp.status_code == 503
        assert "Traceback" not in str(resp.json())


class TestRequestId:
    def test_request_id_present(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("X-Request-ID")

    def test_request_id_echoes_incoming(self, client):
        resp = client.get("/api/health", headers={"X-Request-ID": "abc-123"})
        assert resp.headers["X-Request-ID"] == "abc-123"

    def test_error_response_includes_request_id(self, client, use_unavailable):
        resp = client.get("/api/ready")
        assert resp.headers.get("X-Request-ID") == resp.json()["error"]["request_id"]


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        resp = client.get("/api/health")
        headers = resp.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Referrer-Policy") == "same-origin"

    def test_no_server_header(self, client):
        resp = client.get("/api/health")
        assert "server" not in {k.lower() for k in resp.headers} or not resp.headers.get("Server")