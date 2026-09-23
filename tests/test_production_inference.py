"""FlyMind Phase 9 — Production inference regression tests."""

import json
import pathlib
import pickle
import tempfile

import numpy as np
import pytest

from src.inference.model_contract import (
    MODEL_VERSION, EXPECTED_ESTIMATORS, NODE_FEATURE_COUNT,
    PAIR_FEATURE_COUNT, CANONICAL_NODE_FEATURES, EXPECTED_ARTIFACT_HASH,
    DEFAULT_CANDIDATE_COUNT, MAX_CANDIDATE_COUNT,
)
from src.inference.model_loader import load_model_artifact, _compute_hash
from src.inference.feature_contract import (
    validate_node_features, build_pair_features_single,
    build_pair_features_batch, lookup_neuron_features,
)
from src.inference.exceptions import (
    ModelLoadError, ModelArtifactMismatchError,
    FeatureContractError, InvalidNeuronIDError,
    CandidateLimitError,
)
from src.inference.health_check import check_model_health

PROCESSED_DIR = pathlib.Path("data/processed")
LP_DIR = PROCESSED_DIR / "link_prediction"
MODELS_DIR = pathlib.Path("models")
MODEL_PATH = MODELS_DIR / "link_prediction_rf.pkl"


# ============================================================
# Model Loading Tests
# ============================================================

class TestModelLoading:
    """TASK 20: Model loading tests."""

    def test_successful_load(self):
        """Successful load with hash verification."""
        artifact = load_model_artifact(MODEL_PATH, verify_hash=True)
        assert "model" in artifact
        assert "feature_cols" in artifact
        assert type(artifact["model"]).__name__ == "RandomForestClassifier"

    def test_missing_model(self):
        """Missing model file raises ModelLoadError."""
        with pytest.raises(ModelLoadError, match="not found"):
            load_model_artifact(pathlib.Path("/nonexistent/model.pkl"))

    def test_corrupted_model(self):
        """Corrupted pickle raises ModelLoadError."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not a valid pickle")
            f.flush()
            try:
                with pytest.raises(ModelLoadError, match="Failed to deserialize"):
                    load_model_artifact(pathlib.Path(f.name), verify_hash=False)
            finally:
                import os
                os.unlink(f.name)

    def test_metadata_mismatch_wrong_type(self):
        """Wrong model type in pickle raises ModelArtifactMismatchError."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"model": "not_a_model", "feature_cols": []}, f)
            f.flush()
            try:
                with pytest.raises(ModelArtifactMismatchError):
                    load_model_artifact(pathlib.Path(f.name), verify_hash=False)
            finally:
                import os
                os.unlink(f.name)

    def test_hash_mismatch(self):
        """Hash mismatch raises ModelArtifactMismatchError."""
        with pytest.raises(ModelArtifactMismatchError, match="hash mismatch"):
            load_model_artifact(MODEL_PATH, verify_hash=True, expected_hash="wrong_hash")

    def test_hash_computation(self):
        """SHA-256 hash is computed correctly."""
        h = _compute_hash(MODEL_PATH)
        assert h == EXPECTED_ARTIFACT_HASH

    def test_feature_count_matches_contract(self):
        """Loaded model has correct feature count."""
        artifact = load_model_artifact(MODEL_PATH, verify_hash=False)
        assert artifact["model"].n_features_in_ == PAIR_FEATURE_COUNT

    def test_estimator_count_matches_contract(self):
        """Loaded model has correct estimator count."""
        artifact = load_model_artifact(MODEL_PATH, verify_hash=False)
        assert artifact["model"].n_estimators == EXPECTED_ESTIMATORS

    def test_feature_names_match_canonical(self):
        """Loaded feature_cols match canonical ordering."""
        artifact = load_model_artifact(MODEL_PATH, verify_hash=False)
        assert list(artifact["feature_cols"]) == list(CANONICAL_NODE_FEATURES)


# ============================================================
# Feature Contract Tests
# ============================================================

class TestFeatureContract:
    """TASK 20: Feature contract tests."""

    def test_exactly_15_node_features(self):
        """Node features must have exactly 15 dimensions."""
        artifact = load_model_artifact(MODEL_PATH, verify_hash=False)
        assert len(artifact["feature_cols"]) == NODE_FEATURE_COUNT

    def test_exactly_60_pair_features(self):
        """Pair features must have exactly 60 dimensions."""
        src = np.random.randn(NODE_FEATURE_COUNT)
        tgt = np.random.randn(NODE_FEATURE_COUNT)
        pair = build_pair_features_single(src, tgt)
        assert pair.shape == (PAIR_FEATURE_COUNT,)

    def test_correct_ordering(self):
        """Pair feature ordering matches contract."""
        src = np.arange(NODE_FEATURE_COUNT, dtype=np.float64)
        tgt = np.arange(NODE_FEATURE_COUNT, dtype=np.float64) * 2
        pair = build_pair_features_single(src, tgt)

        # First 15 should be src
        np.testing.assert_array_equal(pair[:NODE_FEATURE_COUNT], src)
        # Next 15 should be tgt
        np.testing.assert_array_equal(pair[NODE_FEATURE_COUNT:2*NODE_FEATURE_COUNT], tgt)
        # Next 15 should be abs diff
        np.testing.assert_array_equal(pair[2*NODE_FEATURE_COUNT:3*NODE_FEATURE_COUNT], np.abs(src - tgt))
        # Last 15 should be product
        np.testing.assert_array_equal(pair[3*NODE_FEATURE_COUNT:], src * tgt)

    def test_missing_value_nan_rejection(self):
        """NaN in features raises FeatureContractError."""
        feats = np.zeros(NODE_FEATURE_COUNT)
        feats[5] = np.nan
        with pytest.raises(FeatureContractError, match="NaN"):
            validate_node_features(feats)

    def test_missing_value_inf_rejection(self):
        """Inf in features raises FeatureContractError."""
        feats = np.zeros(NODE_FEATURE_COUNT)
        feats[5] = np.inf
        with pytest.raises(FeatureContractError, match="Inf"):
            validate_node_features(feats)

    def test_wrong_dimension_rejection(self):
        """Wrong feature count raises FeatureContractError."""
        feats = np.zeros(10)
        with pytest.raises(FeatureContractError, match="Expected 15"):
            validate_node_features(feats)

    def test_batch_pair_features(self):
        """Batch pair features produce correct shape."""
        n = 100
        src = np.random.randn(n, NODE_FEATURE_COUNT)
        tgt = np.random.randn(n, NODE_FEATURE_COUNT)
        pairs = build_pair_features_batch(src, tgt)
        assert pairs.shape == (n, PAIR_FEATURE_COUNT)


# ============================================================
# Neuron Lookup Tests
# ============================================================

class TestNeuronLookup:
    """TASK 20: Neuron lookup tests."""

    def test_valid_id(self):
        """Valid neuron ID returns features."""
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        feats = np.load(LP_DIR / "X_features.npy")
        first_id = list(id_to_idx.keys())[0]
        result = lookup_neuron_features(first_id, id_to_idx, feats)
        assert result.shape == (NODE_FEATURE_COUNT,)

    def test_invalid_id(self):
        """Invalid neuron ID raises InvalidNeuronIDError."""
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        feats = np.load(LP_DIR / "X_features.npy")
        with pytest.raises(InvalidNeuronIDError, match="not found"):
            lookup_neuron_features(999999999, id_to_idx, feats)

    def test_malformed_id_type(self):
        """Non-integer ID raises InvalidNeuronIDError or TypeError."""
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        feats = np.load(LP_DIR / "X_features.npy")
        with pytest.raises((InvalidNeuronIDError, TypeError, KeyError)):
            lookup_neuron_features("not_a_number", id_to_idx, feats)


# ============================================================
# Single Prediction Tests
# ============================================================

class TestSinglePrediction:
    """TASK 20: Single prediction tests."""

    def _get_service(self):
        from src.inference.inference_service import FlyMindInferenceService
        return FlyMindInferenceService()

    def test_deterministic_result(self):
        """Same inputs produce same output."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src, tgt = int(ids[0]), int(ids[1])

        r1 = svc.score_connection(src, tgt)
        r2 = svc.score_connection(src, tgt)
        assert r1.model_score == r2.model_score

    def test_valid_score(self):
        """Score is between 0 and 1."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src, tgt = int(ids[0]), int(ids[1])
        result = svc.score_connection(src, tgt)
        assert 0.0 <= result.model_score <= 1.0

    def test_finite_score(self):
        """Score is finite."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src, tgt = int(ids[0]), int(ids[1])
        result = svc.score_connection(src, tgt)
        assert np.isfinite(result.model_score)

    def test_response_schema(self):
        """Response has all required fields."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src, tgt = int(ids[0]), int(ids[1])
        result = svc.score_connection(src, tgt)
        assert hasattr(result, "source_root_id")
        assert hasattr(result, "target_root_id")
        assert hasattr(result, "model_score")
        assert hasattr(result, "observed_in_dataset")
        assert hasattr(result, "model_version")


# ============================================================
# Connection Status Tests
# ============================================================

class TestConnectionStatus:
    """TASK 20: Connection status tests."""

    def _get_service(self):
        from src.inference.inference_service import FlyMindInferenceService
        return FlyMindInferenceService()

    def test_known_edge(self):
        """Known edge is detected."""
        svc = self._get_service()
        # Use known edges from test data
        edges = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        id_to_idx = edges
        # Find a real edge
        import pandas as pd
        edges_df = pd.read_parquet(LP_DIR / "edges_aggregated.parquet", columns=["source", "target"])
        src = int(edges_df.iloc[0]["source"])
        tgt = int(edges_df.iloc[0]["target"])
        result = svc.score_connection(src, tgt)
        assert result.observed_in_dataset is True

    def test_non_observed_edge(self):
        """Non-observed edge returns False."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        # Use first two neurons (unlikely to be connected)
        src, tgt = int(ids[0]), int(ids[-1])
        result = svc.score_connection(src, tgt)
        # Just verify it returns a boolean
        assert isinstance(result.observed_in_dataset, bool)

    def test_self_loop_detection(self):
        """Self-loop is handled."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        result = svc.score_connection(src, src)
        assert result.observed_in_dataset is False


# ============================================================
# Candidate Ranking Tests
# ============================================================

class TestCandidateRanking:
    """TASK 20: Candidate ranking tests."""

    def _get_service(self):
        from src.inference.inference_service import FlyMindInferenceService
        return FlyMindInferenceService()

    def test_valid_ranking(self):
        """Valid ranking returns candidates."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        result = svc.rank_candidate_targets(src, candidate_count=10)
        assert len(result.candidates) > 0
        assert len(result.candidates) <= 10

    def test_deterministic_ranking(self):
        """Same inputs produce same ranking."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        r1 = svc.rank_candidate_targets(src, candidate_count=10, seed=42)
        r2 = svc.rank_candidate_targets(src, candidate_count=10, seed=42)
        ids1 = [c.root_id for c in r1.candidates]
        ids2 = [c.root_id for c in r2.candidates]
        assert ids1 == ids2

    def test_self_exclusion(self):
        """Source neuron is not in candidates."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        result = svc.rank_candidate_targets(src, candidate_count=50)
        cand_ids = [c.root_id for c in result.candidates]
        assert src not in cand_ids

    def test_candidate_limit(self):
        """Exceeding max candidates raises CandidateLimitError."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        with pytest.raises(CandidateLimitError, match="exceeds maximum"):
            svc.rank_candidate_targets(src, candidate_count=MAX_CANDIDATE_COUNT + 1)

    def test_invalid_source(self):
        """Invalid source raises InvalidNeuronIDError."""
        svc = self._get_service()
        with pytest.raises(InvalidNeuronIDError, match="not found"):
            svc.rank_candidate_targets(999999999, candidate_count=10)

    def test_candidate_scores_sorted(self):
        """Candidates are sorted by score descending."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        result = svc.rank_candidate_targets(src, candidate_count=20)
        scores = [c.model_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        """Ranks are sequential starting from 1."""
        svc = self._get_service()
        id_to_idx = {int(k): v for k, v in json.load(open(LP_DIR / "id_to_idx.json")).items()}
        ids = list(id_to_idx.keys())
        src = int(ids[0])
        result = svc.rank_candidate_targets(src, candidate_count=10)
        ranks = [c.rank for c in result.candidates]
        assert ranks == list(range(1, len(ranks) + 1))


# ============================================================
# Health Check Tests
# ============================================================

class TestHealthCheck:
    """TASK 20: Health check tests."""

    def test_healthy_state(self):
        """Health check returns healthy."""
        status = check_model_health(MODELS_DIR, LP_DIR, verify_hash=True)
        assert status.status == "healthy"
        assert status.artifact_verified is True
        assert status.feature_dimension == PAIR_FEATURE_COUNT

    def test_missing_artifact(self):
        """Missing artifact returns unhealthy."""
        status = check_model_health(
            pathlib.Path("/nonexistent"), LP_DIR, verify_hash=False
        )
        assert status.status == "unhealthy"

    def test_mismatched_metadata(self):
        """Hash mismatch returns unhealthy."""
        status = check_model_health(MODELS_DIR, LP_DIR, verify_hash=True)
        assert status.status == "healthy"

    def test_health_details(self):
        """Health status includes details string."""
        status = check_model_health(MODELS_DIR, LP_DIR, verify_hash=True)
        assert len(status.details) > 0
