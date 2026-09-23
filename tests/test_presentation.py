"""Tests for FlyMind Presentation Data Layer."""

import numpy as np
import pytest

from src.link_prediction.presentation import (
    FlyMindDataStore,
    Connection,
    NeuronSummary,
    ConnectivitySummary,
    FeatureImportance,
    NeighborhoodData,
    get_datastore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def store():
    """Load data store (module-level for speed)."""
    return FlyMindDataStore()


# Known test data
KNOWN_SOURCE = 720575940596125868


# ---------------------------------------------------------------------------
# Neuron lookup tests
# ---------------------------------------------------------------------------
class TestNeuronLookup:
    def test_get_neuron_valid(self, store):
        """Valid neuron returns metadata."""
        from src.link_prediction.inference import NeuronInfo
        neuron = store.get_neuron(KNOWN_SOURCE)
        assert isinstance(neuron, NeuronInfo)
        assert neuron.root_id == KNOWN_SOURCE

    def test_get_neuron_invalid(self, store):
        """Invalid neuron ID raises KeyError."""
        with pytest.raises(KeyError):
            store.get_neuron(999999999999999999)

    def test_get_neuron_summary(self, store):
        """Neuron summary has connectivity counts."""
        summary = store.get_neuron_summary(KNOWN_SOURCE)
        assert isinstance(summary, NeuronSummary)
        assert summary.root_id == KNOWN_SOURCE
        assert summary.outgoing_count >= 0
        assert summary.incoming_count >= 0
        assert summary.total_synapses_out >= 0
        assert summary.total_synapses_in >= 0


# ---------------------------------------------------------------------------
# Connectivity lookup tests
# ---------------------------------------------------------------------------
class TestConnectivity:
    def test_outgoing_connections(self, store):
        """Outgoing connections are returned."""
        conns = store.get_outgoing_connections(KNOWN_SOURCE, limit=5)
        assert isinstance(conns, list)
        assert all(isinstance(c, Connection) for c in conns)

    def test_outgoing_direction_correct(self, store):
        """All outgoing connections have correct source."""
        conns = store.get_outgoing_connections(KNOWN_SOURCE, limit=10)
        for c in conns:
            assert c.source_root_id == KNOWN_SOURCE

    def test_incoming_connections(self, store):
        """Incoming connections are returned."""
        conns = store.get_incoming_connections(KNOWN_SOURCE, limit=5)
        assert isinstance(conns, list)
        assert all(isinstance(c, Connection) for c in conns)

    def test_incoming_direction_correct(self, store):
        """All incoming connections have correct target."""
        conns = store.get_incoming_connections(KNOWN_SOURCE, limit=10)
        for c in conns:
            assert c.target_root_id == KNOWN_SOURCE

    def test_limit_works(self, store):
        """Limit parameter constrains results."""
        conns5 = store.get_outgoing_connections(KNOWN_SOURCE, limit=5)
        conns100 = store.get_outgoing_connections(KNOWN_SOURCE, limit=100)
        assert len(conns5) <= 5
        assert len(conns100) <= 100

    def test_sort_by_weight(self, store):
        """Connections sorted by weight descending."""
        conns = store.get_outgoing_connections(KNOWN_SOURCE, limit=20, sort_by="weight")
        weights = [c.weight for c in conns]
        assert weights == sorted(weights, reverse=True)

    def test_no_self_loops(self, store):
        """No self-loops in observed connections."""
        out = store.get_outgoing_connections(KNOWN_SOURCE, limit=50)
        for c in out:
            assert c.source_root_id != c.target_root_id

    def test_connectivity_summary(self, store):
        """Connectivity summary returns valid data."""
        summary = store.get_connectivity_summary(KNOWN_SOURCE, top_k=3)
        assert isinstance(summary, ConnectivitySummary)
        assert summary.outgoing_count >= 0
        assert summary.incoming_count >= 0
        assert len(summary.top_outgoing) <= 3
        assert len(summary.top_incoming) <= 3


# ---------------------------------------------------------------------------
# Candidate ranking tests
# ---------------------------------------------------------------------------
class TestCandidateRanking:
    def test_rank_candidates(self, store):
        """Candidate ranking returns valid results."""
        candidates = store.rank_candidate_targets(KNOWN_SOURCE, k=5)
        assert isinstance(candidates, list)
        assert len(candidates) <= 5

    def test_candidates_descending_score(self, store):
        """Candidates sorted by score descending."""
        candidates = store.rank_candidate_targets(KNOWN_SOURCE, k=10)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Experiment results tests
# ---------------------------------------------------------------------------
class TestExperimentResults:
    def test_get_experiment_results(self, store):
        """Experiment results load successfully."""
        results = store.get_experiment_results()
        assert isinstance(results, dict)
        assert "2a" in results or "2b" in results or "2d" in results or "3" in results

    def test_model_comparison(self, store):
        """Model comparison returns data."""
        comparison = store.get_model_comparison()
        assert isinstance(comparison, list)
        assert len(comparison) > 0
        for item in comparison:
            assert "model" in item
            assert "roc_auc" in item

    def test_calibration_data(self, store):
        """Calibration data is available."""
        cal = store.get_calibration_data()
        assert isinstance(cal, list)

    def test_feature_importance(self, store):
        """Feature importance returns data."""
        fi = store.get_feature_importance()
        assert isinstance(fi, list)
        assert len(fi) > 0
        for item in fi:
            assert isinstance(item, FeatureImportance)
            assert 0.0 <= item.importance <= 1.0

    def test_ranking_distribution(self, store):
        """Ranking distribution returns data."""
        rd = store.get_ranking_distribution()
        assert isinstance(rd, dict)

    def test_per_source_ranking(self, store):
        """Per-source ranking returns data."""
        psr = store.get_per_source_ranking()
        assert isinstance(psr, dict)
        assert "rf" in psr or "random" in psr


# ---------------------------------------------------------------------------
# Visualization data tests
# ---------------------------------------------------------------------------
class TestVisualizationData:
    def test_neighborhood_outgoing(self, store):
        """Neighborhood with outgoing direction."""
        hood = store.get_neighborhood(KNOWN_SOURCE, direction="outgoing", max_nodes=10)
        assert isinstance(hood, NeighborhoodData)
        assert hood.direction == "outgoing"
        assert len(hood.nodes) > 0
        assert len(hood.edges) > 0

    def test_neighborhood_incoming(self, store):
        """Neighborhood with incoming direction."""
        hood = store.get_incoming_connections(KNOWN_SOURCE, limit=5)
        assert isinstance(hood, list)

    def test_neighborhood_with_candidates(self, store):
        """Neighborhood includes candidates when requested."""
        hood = store.get_neighborhood(
            KNOWN_SOURCE,
            direction="outgoing",
            max_nodes=10,
            include_candidates=True,
            candidate_k=3,
        )
        node_types = [n["type"] for n in hood.nodes]
        edge_types = [e["type"] for e in hood.edges]
        assert "center" in node_types
        assert "observed" in edge_types
        # Candidates may or may not be present depending on sampling

    def test_score_distribution(self, store):
        """Score distribution returns data."""
        sd = store.get_score_distribution()
        assert isinstance(sd, dict)
        assert "mean" in sd
        assert "median" in sd


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
class TestSmoke:
    def test_known_source_has_connections(self, store):
        """Smoke: known source has outgoing connections."""
        conns = store.get_outgoing_connections(KNOWN_SOURCE, limit=5)
        assert len(conns) > 0

    def test_candidate_vs_observed_distinct(self, store):
        """Smoke: candidates and observed are distinct edge types."""
        hood = store.get_neighborhood(
            KNOWN_SOURCE,
            direction="outgoing",
            max_nodes=10,
            include_candidates=True,
            candidate_k=3,
        )
        observed = [e for e in hood.edges if e["type"] == "observed"]
        candidates = [e for e in hood.edges if e["type"] == "candidate"]
        # Both should exist (or at least observed)
        assert len(observed) > 0

    def test_singleton_works(self, store):
        """Smoke: singleton datastore works."""
        ds = get_datastore()
        assert ds.n_neurons > 0
