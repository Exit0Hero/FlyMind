"""Shared fixtures for FlyMind API tests.

Tests use a stubbed ML service so they never load the ~177 MB model artifact.
The stub mirrors the interface the routers depend on (see app.services.ml_service).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/

# Force backend/ to the front of sys.path. PYTHONPATH may already list it, but
# pytest can prepend the repo root (which contains a legacy `app/` package)
# above it; that legacy package would shadow backend/app, so we remove any
# pre-existing entry and pin backend at the head.
if str(BACKEND_DIR) in sys.path:
    sys.path.remove(str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR))

import sys as _sys
# If a legacy `app` package was imported before we could pin the path, drop it
# from sys.modules so this import resolves to backend/app.
_sys.modules.pop("app", None)
from app.main import app  # noqa: E402
from app.services.ml_service import ml_service  # noqa: E402


class FakeInference:
    """Stand-in for FlyMindInference exposing only what routers use."""

    def __init__(self):
        self._rf = SimpleNamespace(n_estimators=500, classes_=[0, 1])
        self._feature_cols = [
            "nt_type_score", "da_avg", "ser_avg", "gaba_avg", "glut_avg",
            "ach_avg", "oct_avg", "length_nm", "area_nm", "size_nm",
            "coord_x", "coord_y", "coord_z", "flow_enc", "side_x_enc",
        ]
        self.n_neurons = 139255
        self.n_edges = 3732460
        self.feature_dim = 60
        self.node_feature_dim = 15
        self.behavior: dict = {"missing": [], "exception": []}
        self._nt_lookup = {
            1: {"name": "neuron-1", "nt_type": "acetylcholinergic", "super_class": "A", "primary_type": "X"},
            2: {"name": "neuron-2", "nt_type": "GABAergic", "super_class": "B", "primary_type": "Y"},
        }

    def get_neuron(self, root_id: int):
        if root_id in self.behavior["missing"]:
            raise KeyError("not found")
        if root_id in self.behavior["exception"]:
            raise RuntimeError("boom")
        return SimpleNamespace(
            root_id=root_id, nt_type="acetylcholinergic", nt_type_score=0.9,
            primary_type="X", super_class="A", flow="out",
            coord_x=1.0, coord_y=2.0, coord_z=3.0,
            length_nm=4.0, area_nm=5.0, size_nm=6.0, side="right", name="neuron-a",
        )

    def score_connection(self, source_root_id: int, target_root_id: int):
        if source_root_id in self.behavior["missing"] or target_root_id in self.behavior["missing"]:
            raise KeyError("not found")
        if source_root_id in self.behavior["exception"] or target_root_id in self.behavior["exception"]:
            raise RuntimeError("boom")
        return SimpleNamespace(
            source_root_id=source_root_id, target_root_id=target_root_id,
            score=0.42, is_known_edge=False, is_self_loop=source_root_id == target_root_id,
        )

    def rank_candidate_targets(self, source_root_id: int, k: int = 10, candidate_pool_size: int = 1000):
        if source_root_id in self.behavior["missing"]:
            raise KeyError("not found")
        if source_root_id in self.behavior["exception"]:
            raise RuntimeError("boom")
        return [
            SimpleNamespace(
                rank=i + 1, source_root_id=source_root_id, target_root_id=100 + i,
                score=0.95 - i * 0.05, target_nt_type="acetylcholinergic",
                target_super_class="A", target_primary_type="X",
            )
            for i in range(k)
        ]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def use_stub(monkeypatch):
    """Point the shared ML service at a controllable fake inference."""
    state = {"fake": FakeInference()}
    monkeypatch.setattr(ml_service, "_loaded", True)
    monkeypatch.setattr(ml_service, "_error", None)
    monkeypatch.setattr(ml_service, "_inference", state["fake"])
    yield state["fake"]


@pytest.fixture
def use_unavailable(monkeypatch):
    """Simulate a failed model load (503 paths)."""
    monkeypatch.setattr(ml_service, "_loaded", False)
    monkeypatch.setattr(ml_service, "_error", "ML model could not be loaded on this instance")
    monkeypatch.setattr(ml_service, "_inference", None)
    yield


def error_body(resp):
    return resp.json()["error"]
