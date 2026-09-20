"""FlyMind Presentation Data Layer — Efficient data access for dashboard consumption.

This module provides a lightweight, efficient interface for:
- Neuron lookup with connection counts
- Observed connectivity retrieval (incoming/outgoing)
- Candidate ranking via the validated RF model
- Research results and feature importance
- Visualization data preparation (neighborhoods, charts)

All outputs preserve the distinction between:
- OBSERVED CONNECTIONS (from the connectome)
- MODEL-SUGGESTED CANDIDATES (hypotheses from the RF model)

Usage:
    store = FlyMindDataStore()
    neuron = store.get_neuron(root_id)
    outgoing = store.get_outgoing_connections(root_id, limit=10)
    candidates = store.rank_candidate_targets(root_id, k=10)
    results = store.get_experiment_results()
"""

import gc
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.link_prediction.config import (
    PROJECT_ROOT, MODELS_DIR, PROCESSED_DIR, LP_DIR, REPORTS_DIR,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, SEED,
)
from src.link_prediction.inference import (
    FlyMindInference, NeuronInfo, ConnectionScore, CandidateTarget,
    get_inference,
)


# ---------------------------------------------------------------------------
# Data classes for presentation
# ---------------------------------------------------------------------------
@dataclass
class Connection:
    """An observed directed connection."""
    source_root_id: int
    target_root_id: int
    weight: int
    num_neuropils: int = 1
    target_nt_type: Optional[str] = None
    target_super_class: Optional[str] = None
    source_nt_type: Optional[str] = None
    source_super_class: Optional[str] = None


@dataclass
class NeuronSummary:
    """Summary of a neuron's connectivity."""
    root_id: int
    name: Optional[str]
    super_class: Optional[str]
    nt_type: Optional[str]
    outgoing_count: int
    incoming_count: int
    total_synapses_out: int
    total_synapses_in: int


@dataclass
class ConnectivitySummary:
    """Full connectivity summary for a neuron."""
    neuron: NeuronInfo
    outgoing_count: int
    incoming_count: int
    total_synapses_out: int
    total_synapses_in: int
    top_outgoing: list
    top_incoming: list


@dataclass
class FeatureImportance:
    """Feature importance from the validated RF model."""
    feature: str
    importance: float
    group: str


@dataclass
class NeighborhoodData:
    """Neighborhood data for network visualization."""
    nodes: list
    edges: list
    source_root_id: int
    direction: str


# ---------------------------------------------------------------------------
# Feature group mapping
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "nt_type_score": "neurotransmitter",
    "da_avg": "neurotransmitter",
    "ser_avg": "neurotransmitter",
    "gaba_avg": "neurotransmitter",
    "glut_avg": "neurotransmitter",
    "ach_avg": "neurotransmitter",
    "oct_avg": "neurotransmitter",
    "length_nm": "morphology",
    "area_nm": "morphology",
    "size_nm": "morphology",
    "coord_x": "spatial",
    "coord_y": "spatial",
    "coord_z": "spatial",
    "flow": "classification",
    "side_x": "spatial",
}


# ---------------------------------------------------------------------------
# FlyMindDataStore — main presentation interface
# ---------------------------------------------------------------------------
class FlyMindDataStore:
    """Presentation data layer for FlyMind.

    Provides efficient access to neuron data, connectivity, research results,
    and visualization data. Wraps the inference layer and adds presentation-
    specific functionality.

    This is a read-only data layer. It does not modify any models or results.
    """

    def __init__(
        self,
        inference: Optional[FlyMindInference] = None,
        data_dir: Optional[Path] = None,
    ):
        """Initialize the data store.

        Args:
            inference: FlyMindInference instance. Uses singleton if None.
            data_dir: Path to processed LP data. Defaults to config value.
        """
        self._inference = inference or get_inference()
        self._data_dir = data_dir or LP_DIR

        # Lazy-loaded attributes
        self._edges_df = None
        self._outgoing_index = None
        self._incoming_index = None
        self._experiment_results = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load connectivity and results on first use."""
        if self._loaded:
            return
        self._load_edges()
        self._load_experiment_results()
        self._loaded = True

    def _load_edges(self) -> None:
        """Load edges and build adjacency indices."""
        edges_path = self._data_dir / "edges_aggregated.parquet"
        self._edges_df = pd.read_parquet(str(edges_path))

        self._outgoing_index = defaultdict(list)
        self._incoming_index = defaultdict(list)

        for _, row in self._edges_df.iterrows():
            src = int(row["source"])
            tgt = int(row["target"])
            w = int(row["weight"])
            n = int(row.get("num_neuropils", 1))
            self._outgoing_index[src].append((tgt, w, n))
            self._incoming_index[tgt].append((src, w, n))

    def _load_experiment_results(self) -> None:
        """Load verified experiment results from JSON files."""
        self._experiment_results = {}

        json_files = {
            "2a": "link_prediction_results.json",
            "2b": "link_prediction_cold_start_results.json",
            "2d": "experiment_2d_results.json",
            "3": "experiment_3_robustness.json",
        }
        for key, fname in json_files.items():
            path = REPORTS_DIR / fname
            if path.exists():
                with open(path) as f:
                    self._experiment_results[key] = json.load(f)

    def _get_nt_info(self, root_id: int) -> dict:
        """Get NT type and super_class for a neuron."""
        try:
            neuron = self._inference.get_neuron(root_id)
            return {
                "nt_type": neuron.nt_type,
                "super_class": neuron.super_class,
            }
        except KeyError:
            return {"nt_type": None, "super_class": None}

    # -------------------------------------------------------------------
    # Neuron lookup
    # -------------------------------------------------------------------
    def get_neuron(self, root_id: int) -> NeuronInfo:
        """Retrieve metadata for a single neuron.

        Args:
            root_id: FlyWire neuron root ID.

        Returns:
            NeuronInfo with available metadata fields.

        Raises:
            KeyError: If root_id is not in the dataset.
        """
        return self._inference.get_neuron(root_id)

    def get_neuron_summary(self, root_id: int) -> NeuronSummary:
        """Get a connectivity summary for a neuron.

        Args:
            root_id: FlyWire neuron root ID.

        Returns:
            NeuronSummary with connection counts and metadata.
        """
        self._ensure_loaded()
        neuron = self._inference.get_neuron(root_id)

        outgoing = self._outgoing_index.get(root_id, [])
        incoming = self._incoming_index.get(root_id, [])

        return NeuronSummary(
            root_id=root_id,
            name=neuron.name,
            super_class=neuron.super_class,
            nt_type=neuron.nt_type,
            outgoing_count=len(outgoing),
            incoming_count=len(incoming),
            total_synapses_out=sum(w for _, w, _ in outgoing),
            total_synapses_in=sum(w for _, w, _ in incoming),
        )

    # -------------------------------------------------------------------
    # Connectivity lookup
    # -------------------------------------------------------------------
    def get_outgoing_connections(
        self,
        root_id: int,
        limit: Optional[int] = None,
        sort_by: str = "weight",
    ) -> list:
        """Get observed outgoing connections from a neuron.

        Args:
            root_id: Presynaptic neuron ID.
            limit: Maximum number of connections to return.
            sort_by: Sort field ("weight" or "target").

        Returns:
            List of Connection objects, sorted by weight descending.
        """
        self._ensure_loaded()
        root_id = int(root_id)

        if root_id not in self._outgoing_index:
            return []

        connections = []
        for tgt, w, n in self._outgoing_index[root_id]:
            nt_info = self._get_nt_info(tgt)
            connections.append(Connection(
                source_root_id=root_id,
                target_root_id=tgt,
                weight=w,
                num_neuropils=n,
                target_nt_type=nt_info["nt_type"],
                target_super_class=nt_info["super_class"],
            ))

        if sort_by == "weight":
            connections.sort(key=lambda c: c.weight, reverse=True)
        elif sort_by == "target":
            connections.sort(key=lambda c: c.target_root_id)

        if limit is not None:
            connections = connections[:limit]

        return connections

    def get_incoming_connections(
        self,
        root_id: int,
        limit: Optional[int] = None,
        sort_by: str = "weight",
    ) -> list:
        """Get observed incoming connections to a neuron.

        Args:
            root_id: Postsynaptic neuron ID.
            limit: Maximum number of connections to return.
            sort_by: Sort field ("weight" or "source").

        Returns:
            List of Connection objects, sorted by weight descending.
        """
        self._ensure_loaded()
        root_id = int(root_id)

        if root_id not in self._incoming_index:
            return []

        connections = []
        for src, w, n in self._incoming_index[root_id]:
            nt_info = self._get_nt_info(src)
            connections.append(Connection(
                source_root_id=src,
                target_root_id=root_id,
                weight=w,
                num_neuropils=n,
                source_nt_type=nt_info["nt_type"],
                source_super_class=nt_info["super_class"],
            ))

        if sort_by == "weight":
            connections.sort(key=lambda c: c.weight, reverse=True)
        elif sort_by == "source":
            connections.sort(key=lambda c: c.source_root_id)

        if limit is not None:
            connections = connections[:limit]

        return connections

    def get_connectivity_summary(
        self,
        root_id: int,
        top_k: int = 5,
    ) -> ConnectivitySummary:
        """Get a full connectivity summary for a neuron.

        Args:
            root_id: Neuron ID.
            top_k: Number of top connections to include.

        Returns:
            ConnectivitySummary with counts and top connections.
        """
        self._ensure_loaded()
        neuron = self._inference.get_neuron(root_id)

        outgoing = self.get_outgoing_connections(root_id, limit=top_k)
        incoming = self.get_incoming_connections(root_id, limit=top_k)

        all_out = self._outgoing_index.get(root_id, [])
        all_in = self._incoming_index.get(root_id, [])

        return ConnectivitySummary(
            neuron=neuron,
            outgoing_count=len(all_out),
            incoming_count=len(all_in),
            total_synapses_out=sum(w for _, w, _ in all_out),
            total_synapses_in=sum(w for _, w, _ in all_in),
            top_outgoing=outgoing,
            top_incoming=incoming,
        )

    # -------------------------------------------------------------------
    # Candidate ranking (delegates to inference layer)
    # -------------------------------------------------------------------
    def rank_candidate_targets(
        self,
        source_root_id: int,
        k: int = 10,
        candidate_pool_size: int = 1000,
        seed: int = SEED,
    ) -> list:
        """Rank model-suggested candidate targets for a source neuron.

        All outputs are HYPOTHESES, not confirmed connections.

        Args:
            source_root_id: Presynaptic neuron ID.
            k: Number of top candidates to return.
            candidate_pool_size: Candidates to sample before filtering.
            seed: Random seed for reproducibility.

        Returns:
            List of CandidateTarget objects.
        """
        return self._inference.rank_candidate_targets(
            source_root_id,
            k=k,
            candidate_pool_size=candidate_pool_size,
            seed=seed,
        )

    # -------------------------------------------------------------------
    # Experiment results
    # -------------------------------------------------------------------
    def get_experiment_results(self) -> dict:
        """Get all verified experiment results.

        Returns:
            Dict with keys: 2a, 2b, 2d, 3 (and their sub-results).
        """
        self._ensure_loaded()
        return dict(self._experiment_results)

    def get_model_comparison(self) -> list:
        """Get model comparison data for visualization.

        Returns:
            List of dicts with model, metric, value.
        """
        self._ensure_loaded()
        results = []

        # Cold-start results (Experiment 2B)
        cs = self._experiment_results.get("2b", {})
        for model_name in ["rf_node_only", "rf_node_plus_heuristic", "gnn", "random"]:
            if model_name in cs:
                m = cs[model_name]
                display_name = {
                    "rf_node_only": "RF Node Features",
                    "rf_node_plus_heuristic": "RF + Heuristics",
                    "gnn": "GraphSAGE",
                    "random": "Random Baseline",
                }.get(model_name, model_name)
                results.append({
                    "model": display_name,
                    "roc_auc": m.get("roc_auc"),
                    "pr_auc": m.get("pr_auc"),
                    "protocol": "cold-start",
                })

        return results

    def get_calibration_data(self) -> list:
        """Get calibration curve data for visualization.

        Returns:
            List of dicts with bin, mean_predicted, observed_rate.
        """
        self._ensure_loaded()
        e3 = self._experiment_results.get("3", {})
        cal = e3.get("3d_calibration", {})
        return cal.get("calibration_curve", [])

    def get_feature_importance(self) -> list:
        """Get RF feature importance for visualization.

        Returns:
            List of FeatureImportance objects.
        """
        self._ensure_loaded()
        e2d = self._experiment_results.get("2d", {})
        top10 = e2d.get("feature_importance_top10", [])

        results = []
        for name, imp in top10:
            group = FEATURE_GROUPS.get(name, "unknown")
            results.append(FeatureImportance(
                feature=name,
                importance=float(imp),
                group=group,
            ))
        return results

    def get_feature_vector(self, root_id: int):
        """Get the 15-dimensional base feature vector for a neuron.

        Args:
            root_id: Neuron root ID.

        Returns:
            numpy array of shape (15,) or None if neuron not found.
        """
        self._ensure_loaded()
        idx = self._id_to_idx.get(root_id)
        if idx is None:
            return None
        return self._X_features[idx]

    def get_ranking_distribution(self) -> dict:
        """Get held-out positive rank distribution (Experiment 3F).

        Returns:
            Dict with ranking metrics.
        """
        self._ensure_loaded()
        e3 = self._experiment_results.get("3", {})
        return e3.get("3f_rank_distribution", {})

    def get_per_source_ranking(self) -> dict:
        """Get per-source ranking metrics (Experiment 3A).

        Returns:
            Dict with RF and random baseline metrics.
        """
        self._ensure_loaded()
        e3 = self._experiment_results.get("3", {})
        return {
            "rf": e3.get("3a_per_source_ranking", {}),
            "random": e3.get("3a_random_baseline", {}),
        }

    # -------------------------------------------------------------------
    # Visualization data
    # -------------------------------------------------------------------
    def get_neighborhood(
        self,
        root_id: int,
        direction: str = "outgoing",
        max_nodes: int = 50,
        include_candidates: bool = False,
        candidate_k: int = 10,
    ) -> NeighborhoodData:
        """Get neighborhood data for network visualization.

        Args:
            root_id: Center neuron ID.
            direction: "outgoing", "incoming", or "both".
            max_nodes: Maximum nodes to include.
            include_candidates: Whether to include model-suggested candidates.
            candidate_k: Number of candidates per source if included.

        Returns:
            NeighborhoodData with nodes and edges for visualization.
        """
        self._ensure_loaded()

        nodes = []
        edges = []
        node_ids = set()

        # Add center node
        neuron = self._inference.get_neuron(root_id)
        nodes.append({
            "id": root_id,
            "label": neuron.name or str(root_id),
            "type": "center",
            "super_class": neuron.super_class,
            "nt_type": neuron.nt_type,
        })
        node_ids.add(root_id)

        # Get observed connections
        if direction in ("outgoing", "both"):
            out_conns = self.get_outgoing_connections(root_id, limit=max_nodes)
            for conn in out_conns:
                if conn.target_root_id not in node_ids and len(nodes) < max_nodes:
                    nt_info = self._get_nt_info(conn.target_root_id)
                    nodes.append({
                        "id": conn.target_root_id,
                        "label": str(conn.target_root_id),
                        "type": "observed_target",
                        "super_class": nt_info["super_class"],
                        "nt_type": nt_info["nt_type"],
                    })
                    node_ids.add(conn.target_root_id)
                edges.append({
                    "source": root_id,
                    "target": conn.target_root_id,
                    "type": "observed",
                    "weight": conn.weight,
                })

        if direction in ("incoming", "both"):
            in_conns = self.get_incoming_connections(root_id, limit=max_nodes)
            for conn in in_conns:
                if conn.source_root_id not in node_ids and len(nodes) < max_nodes:
                    nt_info = self._get_nt_info(conn.source_root_id)
                    nodes.append({
                        "id": conn.source_root_id,
                        "label": str(conn.source_root_id),
                        "type": "observed_source",
                        "super_class": nt_info["super_class"],
                        "nt_type": nt_info["nt_type"],
                    })
                    node_ids.add(conn.source_root_id)
                edges.append({
                    "source": conn.source_root_id,
                    "target": root_id,
                    "type": "observed",
                    "weight": conn.weight,
                })

        # Add candidates if requested
        if include_candidates:
            candidates = self.rank_candidate_targets(root_id, k=candidate_k)
            for cand in candidates:
                if cand.target_root_id not in node_ids and len(nodes) < max_nodes:
                    nt_info = self._get_nt_info(cand.target_root_id)
                    nodes.append({
                        "id": cand.target_root_id,
                        "label": str(cand.target_root_id),
                        "type": "candidate_target",
                        "super_class": nt_info["super_class"],
                        "nt_type": nt_info["nt_type"],
                    })
                    node_ids.add(cand.target_root_id)
                edges.append({
                    "source": root_id,
                    "target": cand.target_root_id,
                    "type": "candidate",
                    "score": cand.score,
                })

        return NeighborhoodData(
            nodes=nodes,
            edges=edges,
            source_root_id=root_id,
            direction=direction,
        )

    def get_score_distribution(self) -> dict:
        """Get candidate score distribution for histogram visualization.

        Returns:
            Dict with score statistics and histogram bins.
        """
        self._ensure_loaded()
        e2d = self._experiment_results.get("2d", {})
        stats = e2d.get("score_distribution", {})
        return {
            "count": stats.get("count", 0),
            "min": stats.get("min", 0),
            "max": stats.get("max", 0),
            "mean": stats.get("mean", 0),
            "median": stats.get("median", 0),
            "std": stats.get("std", 0),
            "p90": stats.get("p90", 0),
            "p95": stats.get("p95", 0),
            "p99": stats.get("p99", 0),
        }

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        """Number of neurons in the dataset."""
        return self._inference.n_neurons

    @property
    def n_edges(self) -> int:
        """Number of observed directed edges."""
        self._ensure_loaded()
        return len(self._edges_df) if self._edges_df is not None else 0


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_datastore_instance: Optional[FlyMindDataStore] = None


def get_datastore() -> FlyMindDataStore:
    """Get or create the singleton data store instance."""
    global _datastore_instance
    if _datastore_instance is None:
        _datastore_instance = FlyMindDataStore()
    return _datastore_instance
