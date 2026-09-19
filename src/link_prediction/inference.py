"""FlyMind Inference Layer — Clean backend interface for connectivity prediction.

This module provides a stable API for:
- Inspecting neuron metadata
- Scoring directed neuron pairs
- Ranking candidate targets for a source neuron

All outputs are model-suggested candidate connections (hypotheses),
NOT biological discoveries or confirmed connections.

Usage:
    model = FlyMindInference()
    neuron = model.get_neuron(root_id)
    score = model.score_connection(source_root_id, target_root_id)
    candidates = model.rank_candidate_targets(source_root_id, k=10)
"""

import gc
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.link_prediction.config import (
    PROJECT_ROOT, MODELS_DIR, PROCESSED_DIR, LP_DIR,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, SEED,
)
from src.link_prediction.negative_sampling import build_sorted_pairs, verify_not_in_edges


# ---------------------------------------------------------------------------
# Data classes for structured return values
# ---------------------------------------------------------------------------
@dataclass
class NeuronInfo:
    """Metadata for a single neuron."""
    root_id: int
    nt_type: Optional[str] = None
    nt_type_score: Optional[float] = None
    primary_type: Optional[str] = None
    super_class: Optional[str] = None
    flow: Optional[str] = None
    coord_x: Optional[float] = None
    coord_y: Optional[float] = None
    coord_z: Optional[float] = None
    length_nm: Optional[float] = None
    area_nm: Optional[float] = None
    size_nm: Optional[float] = None
    side: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ConnectionScore:
    """Score result for a directed pair."""
    source_root_id: int
    target_root_id: int
    score: float
    is_known_edge: bool
    is_self_loop: bool


@dataclass
class CandidateTarget:
    """A ranked candidate target for a source neuron."""
    rank: int
    source_root_id: int
    target_root_id: int
    score: float
    target_nt_type: Optional[str] = None
    target_super_class: Optional[str] = None
    target_primary_type: Optional[str] = None


# ---------------------------------------------------------------------------
# FlyMindInference — main interface
# ---------------------------------------------------------------------------
class FlyMindInference:
    """FlyMind connectivity prediction inference layer.

    Loads the validated RF model and provides clean interfaces for
    neuron inspection, pair scoring, and candidate ranking.

    This is a computational ranking system. All outputs are hypotheses
    to be validated experimentally, NOT confirmed biological connections.
    """

    def __init__(self, model_path: Optional[Path] = None, data_dir: Optional[Path] = None):
        """Initialize the inference layer.

        Args:
            model_path: Path to the RF model pickle. Defaults to models/link_prediction_rf.pkl
            data_dir: Path to processed LP data. Defaults to data/processed/link_prediction/
        """
        self._model_path = model_path or (MODELS_DIR / "link_prediction_rf.pkl")
        self._data_dir = data_dir or LP_DIR

        # Lazy-loaded attributes
        self._rf = None
        self._feature_cols = None
        self._X = None
        self._id_to_idx = None
        self._all_node_ids = None
        self._sorted_edges = None
        self._neuron_table = None
        self._nt_lookup = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load all artifacts on first use."""
        if self._loaded:
            return
        self._load_model()
        self._load_features()
        self._load_edges()
        self._load_neuron_table()
        self._loaded = True

    def _load_model(self) -> None:
        """Load the trained RF model from pickle."""
        with open(self._model_path, "rb") as f:
            data = pickle.load(f)
        self._rf = data["model"]
        self._feature_cols = data["feature_cols"]

    def _load_features(self) -> None:
        """Load node feature matrix and ID-to-index mapping."""
        X_path = self._data_dir / "X_features.npy"
        id_path = self._data_dir / "id_to_idx.json"

        self._X = np.load(str(X_path))
        # Fill NaN values (consistent with training)
        nan_count = int(np.isnan(self._X).sum())
        if nan_count > 0:
            self._X = np.nan_to_num(self._X, nan=0.0, posinf=0.0, neginf=0.0)

        with open(id_path) as f:
            self._id_to_idx = {int(k): v for k, v in json.load(f).items()}
        self._all_node_ids = np.array(sorted(self._id_to_idx.keys()), dtype=np.int64)

    def _load_edges(self) -> None:
        """Load observed edges as compact sorted array for fast membership testing."""
        edges_path = self._data_dir / "edges_aggregated.parquet"
        edges_df = pd.read_parquet(str(edges_path))
        all_edges = edges_df[["source", "target"]].values.astype(np.int64)
        self._sorted_edges = build_sorted_pairs(all_edges[:, 0], all_edges[:, 1])
        del edges_df
        gc.collect()

    def _load_neuron_table(self) -> None:
        """Load neuron metadata table."""
        nt_path = PROCESSED_DIR / "neuron_table.parquet"
        self._neuron_table = pd.read_parquet(str(nt_path))
        # Build lookup dict for fast access
        meta_cols = [
            "root_id", "nt_type", "nt_type_score", "primary_type",
            "super_class", "flow", "coord_x", "coord_y", "coord_z",
            "length_nm", "area_nm", "size_nm", "side_x", "name",
        ]
        available = [c for c in meta_cols if c in self._neuron_table.columns]
        self._nt_lookup = (
            self._neuron_table[available]
            .set_index("root_id")
            .to_dict("index")
        )

    # -------------------------------------------------------------------
    # Public API
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
        self._ensure_loaded()
        root_id = int(root_id)

        if root_id not in self._nt_lookup:
            raise KeyError(f"Neuron {root_id} not found in dataset")

        info = self._nt_lookup[root_id]
        return NeuronInfo(
            root_id=root_id,
            nt_type=info.get("nt_type"),
            nt_type_score=info.get("nt_type_score"),
            primary_type=info.get("primary_type"),
            super_class=info.get("super_class"),
            flow=info.get("flow"),
            coord_x=info.get("coord_x"),
            coord_y=info.get("coord_y"),
            coord_z=info.get("coord_z"),
            length_nm=info.get("length_nm"),
            area_nm=info.get("area_nm"),
            size_nm=info.get("size_nm"),
            side=info.get("side_x"),
            name=info.get("name"),
        )

    def score_connection(
        self,
        source_root_id: int,
        target_root_id: int,
    ) -> ConnectionScore:
        """Score a directed neuron pair for connectivity likelihood.

        Args:
            source_root_id: Presynaptic neuron ID.
            target_root_id: Postsynaptic neuron ID.

        Returns:
            ConnectionScore with RF prediction and edge status.

        Raises:
            KeyError: If either neuron ID is not in the dataset.
        """
        self._ensure_loaded()
        source_root_id = int(source_root_id)
        target_root_id = int(target_root_id)

        # Validate neuron IDs
        if source_root_id not in self._id_to_idx:
            raise KeyError(f"Source neuron {source_root_id} not found in dataset")
        if target_root_id not in self._id_to_idx:
            raise KeyError(f"Target neuron {target_root_id} not found in dataset")

        # Check self-loop
        is_self_loop = source_root_id == target_root_id

        # Check known edge
        is_known = bool(verify_not_in_edges(
            np.array([source_root_id], dtype=np.int64),
            np.array([target_root_id], dtype=np.int64),
            self._sorted_edges,
        )[0] == 0)  # verify_not_in_edges returns True if NOT in edges

        # Build pair features (consistent with training: [x_A, x_B, |x_A-x_B|, x_A*x_B])
        src_idx = self._id_to_idx[source_root_id]
        tgt_idx = self._id_to_idx[target_root_id]
        xa = self._X[src_idx:src_idx + 1]  # (1, 15)
        xb = self._X[tgt_idx:tgt_idx + 1]  # (1, 15)
        pair_features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)

        # RF inference
        score = float(self._rf.predict_proba(pair_features)[0, 1])

        return ConnectionScore(
            source_root_id=source_root_id,
            target_root_id=target_root_id,
            score=score,
            is_known_edge=is_known,
            is_self_loop=is_self_loop,
        )

    def rank_candidate_targets(
        self,
        source_root_id: int,
        k: int = 10,
        candidate_pool_size: int = 1000,
        seed: int = SEED,
    ) -> list[CandidateTarget]:
        """Rank candidate target neurons for a given source.

        Uses the validated RF model to score sampled candidate targets,
        excluding self-loops and known observed edges.

        Args:
            source_root_id: Presynaptic neuron ID to rank targets for.
            k: Number of top candidates to return.
            candidate_pool_size: Number of random candidates to sample before filtering.
            seed: Random seed for reproducibility.

        Returns:
            List of CandidateTarget objects, sorted by score descending.

        Raises:
            KeyError: If source_root_id is not in the dataset.
        """
        self._ensure_loaded()
        source_root_id = int(source_root_id)

        if source_root_id not in self._id_to_idx:
            raise KeyError(f"Source neuron {source_root_id} not found in dataset")

        rng = np.random.default_rng(seed)
        n_nodes = len(self._all_node_ids)

        # Sample candidate targets (more than needed to account for filtering)
        n_sample = min(candidate_pool_size * 3, n_nodes - 1)
        cand_indices = rng.choice(n_nodes, size=n_sample, replace=False)
        cand_ids = self._all_node_ids[cand_indices]

        # Exclude self-loops
        valid = cand_ids != source_root_id
        cand_ids = cand_ids[valid]

        # Exclude known edges
        keep = verify_not_in_edges(
            np.full(len(cand_ids), source_root_id, dtype=np.int64),
            cand_ids,
            self._sorted_edges,
        )
        cand_ids = cand_ids[keep]

        # Take up to candidate_pool_size
        cand_ids = cand_ids[:candidate_pool_size]

        if len(cand_ids) == 0:
            return []

        # Build pair features in chunks
        src_idx = self._id_to_idx[source_root_id]
        src_ids = np.full(len(cand_ids), src_idx, dtype=np.int64)
        tgt_ids = np.array([self._id_to_idx[int(c)] for c in cand_ids], dtype=np.int64)

        # Chunked feature construction
        chunk_size = 10000
        all_scores = []
        for start in range(0, len(cand_ids), chunk_size):
            end = min(start + chunk_size, len(cand_ids))
            ia = src_ids[start:end]
            ib = tgt_ids[start:end]
            xa = self._X[ia]
            xb = self._X[ib]
            pair_features = np.hstack([xa, xb, np.abs(xa - xb), xa * xb]).astype(np.float32)
            scores = self._rf.predict_proba(pair_features)[:, 1]
            all_scores.append(scores)
            del pair_features, xa, xb

        all_scores = np.concatenate(all_scores)

        # Sort by score descending, take top K
        top_k_indices = np.argsort(-all_scores)[:k]

        # Build results with metadata
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            target_id = int(cand_ids[idx])
            score = float(all_scores[idx])

            # Get target metadata (post-hoc, not used in model)
            target_info = self._nt_lookup.get(target_id, {})
            results.append(CandidateTarget(
                rank=rank,
                source_root_id=source_root_id,
                target_root_id=target_id,
                score=score,
                target_nt_type=target_info.get("nt_type"),
                target_super_class=target_info.get("super_class"),
                target_primary_type=target_info.get("primary_type"),
            ))

        return results

    @property
    def n_neurons(self) -> int:
        """Number of neurons in the dataset."""
        self._ensure_loaded()
        return len(self._all_node_ids)

    @property
    def n_edges(self) -> int:
        """Number of observed directed edges."""
        self._ensure_loaded()
        return len(self._sorted_edges)

    @property
    def feature_dim(self) -> int:
        """Expected pair feature dimensionality (4 * n_node_features)."""
        self._ensure_loaded()
        return self._X.shape[1] * 4

    @property
    def node_feature_dim(self) -> int:
        """Number of node-level features."""
        self._ensure_loaded()
        return self._X.shape[1]


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_inference_instance: Optional[FlyMindInference] = None


def get_inference() -> FlyMindInference:
    """Get or create the singleton inference instance."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = FlyMindInference()
    return _inference_instance
