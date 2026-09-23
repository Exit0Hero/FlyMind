"""FlyMind Production Inference — Core inference service."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .exceptions import (
    InvalidNeuronIDError, InferenceError, ModelLoadError,
    FeatureContractError, CandidateLimitError,
)
from .model_contract import (
    CANONICAL_NODE_FEATURES, NODE_FEATURE_COUNT, PAIR_FEATURE_COUNT,
    MODEL_VERSION, DEFAULT_CANDIDATE_COUNT, MAX_CANDIDATE_COUNT,
    CANDIDATE_POOL_MULTIPLIER,
)
from .model_loader import load_model_artifact
from .feature_contract import (
    build_pair_features_single, lookup_neuron_features,
    validate_node_features,
)

logger = logging.getLogger(__name__)


def _build_sorted_edge_array(edges_df: pd.DataFrame) -> np.ndarray:
    """Build compact sorted int64 edge array for binary-search membership."""
    pairs = edges_df[["source", "target"]].values.astype(np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0].argsort()]
    return pairs


def _verify_not_in_edges(
    sources: np.ndarray, targets: np.ndarray, sorted_edges: np.ndarray
) -> np.ndarray:
    """Vectorized check: returns True where the pair is NOT in sorted_edges."""
    queries = np.column_stack([sources, targets]).astype(np.int64)
    queries = np.sort(queries, axis=1)

    left_idx = np.searchsorted(sorted_edges[:, 0], queries[:, 0], side="left")
    right_idx = np.searchsorted(sorted_edges[:, 0], queries[:, 0], side="right")

    found = np.zeros(len(queries), dtype=bool)
    for i in range(len(queries)):
        if left_idx[i] < right_idx[i]:
            candidates = sorted_edges[left_idx[i]:right_idx[i]]
            mask = candidates[:, 1] == queries[i, 1]
            if mask.any():
                found[i] = True

    return ~found


@dataclass
class ConnectionPrediction:
    """Result of a single pair prediction."""
    source_root_id: int
    target_root_id: int
    model_score: float
    observed_in_dataset: bool
    model_version: str = MODEL_VERSION
    predicted_class: int = 0


@dataclass
class CandidateTarget:
    """A ranked candidate target."""
    rank: int
    root_id: int
    model_score: float
    nt_type: Optional[str] = None
    super_class: Optional[str] = None
    primary_type: Optional[str] = None
    observed_in_dataset: bool = False


@dataclass
class RankingResult:
    """Result of candidate ranking."""
    source_root_id: int
    candidates: List[CandidateTarget]
    model_version: str = MODEL_VERSION
    duration_ms: float = 0.0


class FlyMindInferenceService:
    """Production inference service with full validation and safety controls."""

    def __init__(self, data_root: Optional[Path] = None):
        """Initialize the service.

        Args:
            data_root: Path to processed data directory. Defaults to project standard.
        """
        from ..link_prediction.config import LP_DIR, PROCESSED_DIR, MODELS_DIR

        self._data_root = data_root or PROCESSED_DIR
        self._lp_dir = LP_DIR
        self._models_dir = MODELS_DIR

        # Loaded artifacts (lazy)
        self._model = None
        self._feature_cols = None
        self._feature_matrix = None
        self._id_to_idx = None
        self._all_node_ids = None
        self._sorted_edges = None
        self._neuron_table = None
        self._nt_lookup = None
        self._loaded = False
        self._load_time = 0.0
        self._artifact_hash = None

    def _ensure_loaded(self) -> None:
        """Lazy-load all artifacts on first use."""
        if self._loaded:
            return

        t0 = time.perf_counter()

        # Load model
        model_path = self._models_dir / "link_prediction_rf.pkl"
        artifact = load_model_artifact(model_path)
        self._model = artifact["model"]
        self._feature_cols = artifact["feature_cols"]

        # Load features
        features_path = self._lp_dir / "X_features.npy"
        raw = np.load(features_path, allow_pickle=False)
        self._feature_matrix = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        del raw

        # Load id_to_idx mapping
        id_map_path = self._lp_dir / "id_to_idx.json"
        with open(id_map_path, "r") as f:
            self._id_to_idx = {int(k): v for k, v in json.load(f).items()}
        self._all_node_ids = np.array(list(self._id_to_idx.keys()), dtype=np.int64)

        # Load edges
        edges_path = self._lp_dir / "edges_aggregated.parquet"
        edges_df = pd.read_parquet(edges_path, columns=["source", "target"])
        self._sorted_edges = _build_sorted_edge_array(edges_df)
        del edges_df

        # Load neuron table
        neuron_path = self._data_root / "neuron_table.parquet"
        nt_df = pd.read_parquet(neuron_path, columns=["root_id", "nt_type", "super_class", "primary_type"])
        self._neuron_table = nt_df.set_index("root_id", drop=False)
        self._nt_lookup = {
            row["root_id"]: row.to_dict()
            for _, row in nt_df.iterrows()
        }
        del nt_df

        self._loaded = True
        self._load_time = time.perf_counter() - t0
        logger.info(
            "FlyMind inference loaded in %.3fs: %d neurons, %d edges",
            self._load_time, len(self._id_to_idx), len(self._sorted_edges),
        )

    def get_neuron(self, root_id: int) -> Optional[Dict[str, Any]]:
        """Get neuron metadata by root_id.

        Returns:
            Dict with nt_type, super_class, primary_type, or None if not found.
        """
        self._ensure_loaded()
        if root_id not in self._nt_lookup:
            return None
        return self._nt_lookup[root_id]

    def score_connection(
        self, source_root_id: int, target_root_id: int
    ) -> ConnectionPrediction:
        """Score a single source-target pair.

        Returns:
            ConnectionPrediction with model_score and observed status.

        Raises:
            InvalidNeuronIDError: If either ID is not found.
            InferenceError: If prediction fails.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()

        # Validate IDs exist
        if source_root_id not in self._id_to_idx:
            raise InvalidNeuronIDError(
                f"Source neuron ID {source_root_id} not found in the FlyMind dataset."
            )
        if target_root_id not in self._id_to_idx:
            raise InvalidNeuronIDError(
                f"Target neuron ID {target_root_id} not found in the FlyMind dataset."
            )

        # Look up features
        src_feats = lookup_neuron_features(
            source_root_id, self._id_to_idx, self._feature_matrix
        )
        tgt_feats = lookup_neuron_features(
            target_root_id, self._id_to_idx, self._feature_matrix
        )

        # Build pair features
        pair = build_pair_features_single(src_feats, tgt_feats)

        # Run model
        try:
            proba = self._model.predict_proba(pair.reshape(1, -1))[0]
        except Exception as e:
            raise InferenceError(f"Model prediction failed: {e}") from e

        score = float(proba[1])
        predicted_class = int(np.argmax(proba))

        # Check observed edge
        src_arr = np.array([source_root_id], dtype=np.int64)
        tgt_arr = np.array([target_root_id], dtype=np.int64)
        is_not_observed = _verify_not_in_edges(src_arr, tgt_arr, self._sorted_edges)
        observed = bool(not is_not_observed[0])

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "score_connection(%d, %d): score=%.4f, observed=%s, %.1fms",
            source_root_id, target_root_id, score, observed, elapsed,
        )

        return ConnectionPrediction(
            source_root_id=source_root_id,
            target_root_id=target_root_id,
            model_score=score,
            observed_in_dataset=observed,
            predicted_class=predicted_class,
        )

    def rank_candidate_targets(
        self,
        source_root_id: int,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        *,
        seed: int = 42,
    ) -> RankingResult:
        """Rank candidate targets for a given source neuron.

        Args:
            source_root_id: Source neuron ID.
            candidate_count: Number of candidates to rank (max MAX_CANDIDATE_COUNT).
            seed: Random seed for reproducibility.

        Returns:
            RankingResult with ranked candidates.

        Raises:
            InvalidNeuronIDError: If source ID not found.
            CandidateLimitError: If requested count exceeds limit.
            InferenceError: If prediction fails.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()

        # Validate source
        if source_root_id not in self._id_to_idx:
            raise InvalidNeuronIDError(
                f"Source neuron ID {source_root_id} not found in the FlyMind dataset."
            )

        # Validate candidate count
        if candidate_count < 1:
            raise CandidateLimitError(
                f"Candidate count must be >= 1, got {candidate_count}"
            )
        if candidate_count > MAX_CANDIDATE_COUNT:
            raise CandidateLimitError(
                f"Candidate count {candidate_count} exceeds maximum {MAX_CANDIDATE_COUNT}."
            )

        # Sample candidate pool
        n_nodes = len(self._all_node_ids)
        pool_size = min(candidate_count * CANDIDATE_POOL_MULTIPLIER, n_nodes - 1)
        rng = np.random.default_rng(seed)
        all_ids = self._all_node_ids

        # Exclude source
        mask = all_ids != source_root_id
        eligible_ids = all_ids[mask]

        # Sample candidates
        if len(eligible_ids) <= pool_size:
            cand_ids = eligible_ids.copy()
        else:
            cand_idx = rng.choice(len(eligible_ids), size=pool_size, replace=False)
            cand_ids = eligible_ids[cand_idx]

        # Filter known edges
        is_not_observed = _verify_not_in_edges(
            np.full(len(cand_ids), source_root_id, dtype=np.int64),
            cand_ids,
            self._sorted_edges,
        )
        cand_ids = cand_ids[is_not_observed]

        # Take top candidates
        cand_ids = cand_ids[:candidate_count]

        # Build features in bounded chunks
        chunk_size = 10_000
        all_scores = []

        for start in range(0, len(cand_ids), chunk_size):
            end = min(start + chunk_size, len(cand_ids))
            chunk_ids = cand_ids[start:end]

            src_feats = np.tile(
                self._feature_matrix[self._id_to_idx[source_root_id]],
                (len(chunk_ids), 1)
            )
            tgt_feats = self._feature_matrix[
                [self._id_to_idx[int(c)] for c in chunk_ids]
            ]

            # Build pair features
            pairs = np.concatenate(
                [src_feats, tgt_feats, np.abs(src_feats - tgt_feats), src_feats * tgt_feats],
                axis=1,
            )

            try:
                proba = self._model.predict_proba(pairs)[:, 1]
                all_scores.append(proba)
            except Exception as e:
                raise InferenceError(f"Model prediction failed: {e}") from e

            del pairs, src_feats, tgt_feats

        if all_scores:
            scores = np.concatenate(all_scores)
        else:
            scores = np.array([])

        # Sort by score descending
        if len(scores) > 0:
            top_k = min(candidate_count, len(scores))
            top_indices = np.argsort(-scores)[:top_k]
        else:
            top_indices = np.array([], dtype=int)

        # Build results
        candidates = []
        for rank, idx in enumerate(top_indices, 1):
            root_id = int(cand_ids[idx])
            nt_info = self._nt_lookup.get(root_id, {})
            candidates.append(CandidateTarget(
                rank=rank,
                root_id=root_id,
                model_score=float(scores[idx]),
                nt_type=nt_info.get("nt_type"),
                super_class=nt_info.get("super_class"),
                primary_type=nt_info.get("primary_type"),
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "rank_candidate_targets(%d): %d candidates in %.1fms",
            source_root_id, len(candidates), elapsed,
        )

        return RankingResult(
            source_root_id=source_root_id,
            candidates=candidates,
            duration_ms=elapsed,
        )
