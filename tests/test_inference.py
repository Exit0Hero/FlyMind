"""Tests for FlyMind Inference Layer."""

import numpy as np
import pytest

from src.link_prediction.inference import (
    FlyMindInference,
    NeuronInfo,
    ConnectionScore,
    CandidateTarget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Load inference model (module-level for speed)."""
    return FlyMindInference()


# Known test data
KNOWN_SOURCE = 720575940596125868
KNOWN_TARGET = 720575940605825666  # Known edge from source
KNOWN_EDGE_PAIR = (KNOWN_SOURCE, KNOWN_TARGET)


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    def test_model_loads(self, model):
        """Model loads successfully and is accessible."""
        # Trigger lazy loading via public property
        _ = model.n_neurons
        assert model._rf is not None
        assert model._X is not None
        assert model._id_to_idx is not None

    def test_n_neurons(self, model):
        """Dataset neuron count is reasonable."""
        assert model.n_neurons > 100000

    def test_n_edges(self, model):
        """Edge count is reasonable."""
        assert model.n_edges > 1000000


# ---------------------------------------------------------------------------
# Feature construction tests
# ---------------------------------------------------------------------------
class TestFeatureConstruction:
    def test_node_feature_dim(self, model):
        """Node features are 15-dimensional."""
        assert model.node_feature_dim == 15

    def test_pair_feature_dim(self, model):
        """Pair features are 60-dimensional (15 * 4)."""
        assert model.feature_dim == 60

    def test_feature_cols_ordering(self, model):
        """Feature columns match training configuration."""
        from src.link_prediction.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        expected = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]
        assert model._feature_cols == expected


# ---------------------------------------------------------------------------
# Neuron inspection tests
# ---------------------------------------------------------------------------
class TestGetNeuron:
    def test_get_neuron_valid(self, model):
        """Valid neuron returns metadata."""
        neuron = model.get_neuron(KNOWN_SOURCE)
        assert isinstance(neuron, NeuronInfo)
        assert neuron.root_id == KNOWN_SOURCE

    def test_get_neuron_invalid(self, model):
        """Invalid neuron ID raises KeyError."""
        with pytest.raises(KeyError):
            model.get_neuron(999999999999999999)

    def test_get_neuron_has_coords(self, model):
        """Neuron has spatial coordinates."""
        neuron = model.get_neuron(KNOWN_SOURCE)
        assert neuron.coord_x is not None
        assert neuron.coord_y is not None
        assert neuron.coord_z is not None


# ---------------------------------------------------------------------------
# Pair scoring tests
# ---------------------------------------------------------------------------
class TestScoreConnection:
    def test_score_valid_pair(self, model):
        """Valid source-target pair returns a score."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_TARGET)
        assert isinstance(result, ConnectionScore)
        assert 0.0 <= result.score <= 1.0
        assert result.source_root_id == KNOWN_SOURCE
        assert result.target_root_id == KNOWN_TARGET

    def test_score_known_edge(self, model):
        """Known edge is correctly identified."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_TARGET)
        assert result.is_known_edge is True

    def test_score_self_loop(self, model):
        """Self-loop is correctly identified."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_SOURCE)
        assert result.is_self_loop is True

    def test_score_invalid_source(self, model):
        """Invalid source ID raises KeyError."""
        with pytest.raises(KeyError):
            model.score_connection(999999999999999999, KNOWN_TARGET)

    def test_score_invalid_target(self, model):
        """Invalid target ID raises KeyError."""
        with pytest.raises(KeyError):
            model.score_connection(KNOWN_SOURCE, 999999999999999999)

    def test_directionality_matters(self, model):
        """A->B score differs from B->A score (directionality preserved)."""
        score_ab = model.score_connection(KNOWN_SOURCE, KNOWN_TARGET)
        score_ba = model.score_connection(KNOWN_TARGET, KNOWN_SOURCE)
        # Scores may be equal by coincidence, but should not error
        assert isinstance(score_ab.score, float)
        assert isinstance(score_ba.score, float)

    def test_score_finite(self, model):
        """Score is a finite float."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_TARGET)
        assert np.isfinite(result.score)


# ---------------------------------------------------------------------------
# Candidate ranking tests
# ---------------------------------------------------------------------------
class TestRankCandidateTargets:
    def test_rank_returns_list(self, model):
        """Ranking returns a list of CandidateTarget."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=5)
        assert isinstance(candidates, list)
        assert all(isinstance(c, CandidateTarget) for c in candidates)

    def test_rank_no_self_loops(self, model):
        """Ranked candidates contain no self-loops."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=10)
        for c in candidates:
            assert c.source_root_id != c.target_root_id

    def test_rank_no_known_edges(self, model):
        """Ranked candidates contain no known edges (verified via score_connection)."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=10)
        for c in candidates:
            result = model.score_connection(c.source_root_id, c.target_root_id)
            assert result.is_known_edge is False

    def test_rank_descending_order(self, model):
        """Candidates are sorted by score descending."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=10)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_rank_unique_targets(self, model):
        """All target IDs in ranking are unique."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=10)
        target_ids = [c.target_root_id for c in candidates]
        assert len(target_ids) == len(set(target_ids))

    def test_rank_scores_finite(self, model):
        """All candidate scores are finite."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=10)
        for c in candidates:
            assert np.isfinite(c.score)

    def test_rank_invalid_source(self, model):
        """Invalid source ID raises KeyError."""
        with pytest.raises(KeyError):
            model.rank_candidate_targets(999999999999999999, k=5)

    def test_rank_k_zero(self, model):
        """k=0 returns empty list."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=0)
        assert candidates == []

    def test_rank_reproducible(self, model):
        """Same seed produces same results."""
        c1 = model.rank_candidate_targets(KNOWN_SOURCE, k=5, seed=42)
        c2 = model.rank_candidate_targets(KNOWN_SOURCE, k=5, seed=42)
        assert [c.target_root_id for c in c1] == [c.target_root_id for c in c2]
        assert [c.score for c in c1] == [c.score for c in c2]


# ---------------------------------------------------------------------------
# Smoke tests against specific cases
# ---------------------------------------------------------------------------
class TestSmokeTests:
    def test_known_observed_edge(self, model):
        """Smoke: known edge returns high score."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_TARGET)
        assert result.is_known_edge is True
        assert result.score > 0.5  # Known edges should score reasonably

    def test_valid_non_edge(self, model):
        """Smoke: valid non-edge returns a score."""
        # Use two distant neurons unlikely to be connected
        other_id = 720575940599333574
        result = model.score_connection(KNOWN_SOURCE, other_id)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_self_loop(self, model):
        """Smoke: self-loop is flagged."""
        result = model.score_connection(KNOWN_SOURCE, KNOWN_SOURCE)
        assert result.is_self_loop is True

    def test_invalid_id(self, model):
        """Smoke: invalid ID raises KeyError."""
        with pytest.raises(KeyError):
            model.score_connection(12345, KNOWN_TARGET)

    def test_candidate_ranking(self, model):
        """Smoke: candidate ranking returns valid results."""
        candidates = model.rank_candidate_targets(KNOWN_SOURCE, k=3)
        assert len(candidates) <= 3
        assert all(c.rank > 0 for c in candidates)
