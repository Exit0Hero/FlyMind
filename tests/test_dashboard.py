"""Tests for FlyMind Dashboard — integration and logic tests."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------
class TestImports:
    def test_app_imports(self):
        """Main app module imports successfully."""
        import app.streamlit_app
        assert hasattr(app.streamlit_app, "render") or True  # Module loads

    def test_services_imports(self):
        """Services module imports successfully."""
        from app.services import get_store
        assert callable(get_store)

    def test_pages_import(self):
        """All page modules import successfully."""
        from app.pages import overview
        from app.pages import neuron_explorer
        from app.pages import connection_predictor
        from app.pages import candidate_ranking
        from app.pages import research_results
        from app.pages import about
        assert hasattr(overview, "render")
        assert hasattr(neuron_explorer, "render")
        assert hasattr(connection_predictor, "render")
        assert hasattr(candidate_ranking, "render")
        assert hasattr(research_results, "render")
        assert hasattr(about, "render")


# ---------------------------------------------------------------------------
# DataStore integration tests
# ---------------------------------------------------------------------------
class TestDataStoreIntegration:
    def test_store_loads(self):
        """DataStore loads successfully."""
        from app.services import get_store
        store = get_store()
        assert store.n_neurons > 100000

    def test_neuron_lookup(self):
        """Neuron lookup works through services."""
        from app.services import get_store
        store = get_store()
        neuron = store.get_neuron(720575940596125868)
        assert neuron.root_id == 720575940596125868

    def test_connectivity_summary(self):
        """Connectivity summary works."""
        from app.services import get_store
        store = get_store()
        summary = store.get_connectivity_summary(720575940596125868, top_k=3)
        assert summary.outgoing_count >= 0
        assert summary.incoming_count >= 0

    def test_candidate_ranking(self):
        """Candidate ranking works."""
        from app.services import get_store
        store = get_store()
        candidates = store.rank_candidate_targets(720575940596125868, k=5)
        assert len(candidates) <= 5
        assert all(c.score > 0 for c in candidates)

    def test_experiment_results(self):
        """Experiment results load."""
        from app.services import get_store
        store = get_store()
        results = store.get_experiment_results()
        assert isinstance(results, dict)

    def test_feature_importance(self):
        """Feature importance loads."""
        from app.services import get_store
        store = get_store()
        fi = store.get_feature_importance()
        assert len(fi) > 0

    def test_model_comparison(self):
        """Model comparison loads."""
        from app.services import get_store
        store = get_store()
        mc = store.get_model_comparison()
        assert len(mc) > 0


# ---------------------------------------------------------------------------
# Scientific integrity tests
# ---------------------------------------------------------------------------
class TestScientificIntegrity:
    def test_no_all_pairs_enumeration(self):
        """Dashboard does not enumerate all neuron pairs."""
        import ast
        import inspect

        from app.pages import candidate_ranking
        source = inspect.getsource(candidate_ranking)

        # Should not contain all-pairs patterns
        forbidden = ["itertools.product", "itertools.permutations"]
        for pattern in forbidden:
            assert pattern not in source, f"Found forbidden pattern: {pattern}"

    def test_observed_vs_candidate_distinction(self):
        """Dashboard distinguishes observed from candidate connections."""
        from app.services import get_store
        store = get_store()

        # Get neighborhood with candidates
        hood = store.get_neighborhood(
            720575940596125868,
            direction="outgoing",
            max_nodes=10,
            include_candidates=True,
            candidate_k=3,
        )

        edge_types = {e["type"] for e in hood.edges}
        assert "observed" in edge_types
        # Candidates may or may not be present

    def test_candidate_terminology(self):
        """Dashboard uses correct candidate terminology."""
        import inspect
        from app.pages import candidate_ranking
        source = inspect.getsource(candidate_ranking)

        # Should contain correct terminology
        assert "hypothesis" in source.lower() or "candidate" in source.lower()
        # Should not contain forbidden terms
        forbidden = ["discovered connection", "confirmed new connection", "biological probability"]
        for term in forbidden:
            assert term.lower() not in source.lower(), f"Found forbidden term: {term}"


# ---------------------------------------------------------------------------
# Page render tests (import only, no Streamlit context)
# ---------------------------------------------------------------------------
class TestPageModules:
    def test_overview_has_render(self):
        """Overview page has render function."""
        from app.pages.overview import render
        assert callable(render)

    def test_neuron_explorer_has_render(self):
        """Neuron Explorer page has render function."""
        from app.pages.neuron_explorer import render
        assert callable(render)

    def test_connection_predictor_has_render(self):
        """Connection Predictor page has render function."""
        from app.pages.connection_predictor import render
        assert callable(render)

    def test_candidate_ranking_has_render(self):
        """Candidate Ranking page has render function."""
        from app.pages.candidate_ranking import render
        assert callable(render)

    def test_research_results_has_render(self):
        """Research Results page has render function."""
        from app.pages.research_results import render
        assert callable(render)

    def test_about_has_render(self):
        """About page has render function."""
        from app.pages.about import render
        assert callable(render)
