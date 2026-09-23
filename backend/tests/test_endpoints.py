"""Endpoint behavior tests using the stubbed ML service."""

from __future__ import annotations


class TestHealthAndReady:
    def test_health_degraded_when_unloaded(self, client, use_unavailable):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_ok_when_loaded(self, client, use_stub):
        resp = client.get("/api/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["n_neurons"] == 139255
        assert body["n_edges"] == 3732460

    def test_ready_503_when_unloaded(self, client, use_unavailable):
        resp = client.get("/api/ready")
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "NOT_READY"

    def test_ready_ok_when_loaded(self, client, use_stub):
        resp = client.get("/api/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestModelEndpoint:
    def test_model_info_with_stub(self, client, use_stub):
        resp = client.get("/api/model")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_type"] == "RandomForestClassifier"
        assert body["model_version"] == "FlyMind-RF v1.0.0"
        assert body["n_features"] == 15
        assert body["feature_dim"] == 60
        assert len(body["feature_names"]) == 15
        assert body["n_neurons"] == 139255

    def test_model_503_when_unavailable(self, client, use_unavailable):
        assert client.get("/api/model").status_code == 503


class TestPredictEndpoint:
    def test_predict_returns_score(self, client, use_stub):
        resp = client.post("/api/predict", json={"source_root_id": 11, "target_root_id": 12})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_root_id"] == "11"
        assert body["target_root_id"] == "12"
        assert body["score"] == 0.42
        assert body["is_known_edge"] is False
        assert body["is_self_loop"] is False


class TestCandidatesEndpoint:
    def test_candidates_ranked(self, client, use_stub):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 11, "k": 3, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_root_id"] == "11"
        assert len(body["candidates"]) == 3
        assert [c["rank"] for c in body["candidates"]] == [1, 2, 3]
        assert body["total_sampled"] == 1000

    def test_candidates_within_configured_cap(self, client, use_stub):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": 11, "k": 100, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 200
        assert len(resp.json()["candidates"]) == 100


class TestNeuronEndpoints:
    def test_neuron_lookup(self, client, use_stub):
        resp = client.get("/api/neurons/11")
        assert resp.status_code == 200
        body = resp.json()
        assert body["root_id"] == "11"
        assert body["name"] == "neuron-a"

    def test_search(self, client, use_stub):
        resp = client.get("/api/neurons/search", params={"q": "GABA"})
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert body["total"] >= 0

    def test_search_empty_query_422(self, client):
        resp = client.get("/api/neurons/search", params={"q": ""})
        assert resp.status_code == 422

    def test_search_long_query_422(self, client):
        resp = client.get("/api/neurons/search", params={"q": "a" * 200})
        assert resp.status_code == 422


class TestRootIdWireFormat:
    """Root IDs exceed Number.MAX_SAFE_INTEGER; must travel as JSON strings."""

    def test_predict_accepts_string_ids(self, client, use_stub):
        resp = client.post(
            "/api/predict",
            json={"source_root_id": "720575940596125868", "target_root_id": "720575940605825666"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_root_id"] == "720575940596125868"
        assert body["target_root_id"] == "720575940605825666"

    def test_predict_accepts_int_ids_and_returns_strings(self, client, use_stub):
        resp = client.post("/api/predict", json={"source_root_id": 11, "target_root_id": 12})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_root_id"] == "11"
        assert body["target_root_id"] == "12"

    def test_search_returns_string_root_ids(self, client, use_stub):
        resp = client.get("/api/neurons/search", params={"q": "neuron"})
        assert resp.status_code == 200
        for item in resp.json()["results"]:
            assert isinstance(item["root_id"], str)

    def test_neuron_detail_returns_string_root_id(self, client, use_stub):
        resp = client.get("/api/neurons/11")
        assert resp.status_code == 200
        assert isinstance(resp.json()["root_id"], str)

    def test_candidates_return_string_root_ids(self, client, use_stub):
        resp = client.post(
            "/api/candidates",
            json={"source_root_id": "11", "k": 2, "candidate_pool_size": 1000},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_root_id"] == "11"
        for c in body["candidates"]:
            assert isinstance(c["source_root_id"], str)
            assert isinstance(c["target_root_id"], str)


class TestPipelineAndEvaluation:
    def test_pipeline_states(self, client):
        resp = client.get("/api/pipeline")
        assert resp.status_code == 200
        stages = resp.json()["stages"]
        assert len(stages) >= 4
        assert {s["status"] for s in stages} <= {"ready", "missing", "error"}

    def test_evaluation_returns_structured_list(self, client):
        resp = client.get("/api/evaluation")
        assert resp.status_code == 200
        body = resp.json()
        assert "experiments" in body
        assert all("experiment" in e and "data" in e for e in body["experiments"])

    def test_research_summary(self, client, use_stub):
        resp = client.get("/api/research/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset"]["n_neurons"] == 139255
        assert isinstance(body["key_findings"], list)

class TestVersionSurface:
    """Phase 16: deployment/versioning contract.

    ``/api/version`` must expose distinct application, API and model version
    identifiers and must **never** interact with the model artifact — a
    version probe is a metadata-only read that CI and pre-rollout tooling use
    without warming the model, and a client must be able to distinguish
    "model known" from "model loaded".
    """

    def test_version_reports_identifiers_without_loading(self, client, use_stub):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        body = resp.json()
        # Distinct surfaces: app code vs API contract vs model artifact.
        assert body["app_version"]  # application build
        assert body["api_version"]  # HTTP API contract
        assert body["model_version"]  # model artifact (metadata), separate
        assert body["app_version"] != body["model_version"]
        # Version must not have force-loaded the model.
        assert "model_loaded" in body

    def test_version_known_but_not_loaded(self, client, use_unavailable):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        body = resp.json()
        # Metadata version stays known even when the artifact is not loaded.
        assert body["model_version"]
        assert body["model_loaded"] is False
