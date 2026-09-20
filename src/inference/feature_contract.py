"""FlyMind Production Inference — Feature construction with contract validation."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .exceptions import FeatureContractError, InvalidNeuronIDError
from .model_contract import CANONICAL_NODE_FEATURES, NODE_FEATURE_COUNT, PAIR_FEATURE_COUNT

logger = logging.getLogger(__name__)


def validate_node_features(
    features: np.ndarray,
    *,
    expected_count: int = NODE_FEATURE_COUNT,
    context: str = "",
) -> np.ndarray:
    """Validate a node feature vector against the contract.

    Args:
        features: Feature array of shape (15,) or (n, 15).
        expected_count: Expected feature count.
        context: Description for error messages.

    Returns:
        Validated features as float64.

    Raises:
        FeatureContractError: If features do not match contract.
    """
    features = np.asarray(features, dtype=np.float64)

    if features.ndim == 1:
        if features.shape[0] != expected_count:
            raise FeatureContractError(
                f"{context}: Expected {expected_count} features, got {features.shape[0]}"
            )
    elif features.ndim == 2:
        if features.shape[1] != expected_count:
            raise FeatureContractError(
                f"{context}: Expected {expected_count} features per row, got {features.shape[1]}"
            )
    else:
        raise FeatureContractError(
            f"{context}: Expected 1D or 2D array, got {features.ndim}D"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(features)):
        raise FeatureContractError(
            f"{context}: Feature vector contains NaN values"
        )
    if np.any(np.isinf(features)):
        raise FeatureContractError(
            f"{context}: Feature vector contains Inf values"
        )

    return features


def build_pair_features_single(
    src_features: np.ndarray,
    tgt_features: np.ndarray,
) -> np.ndarray:
    """Build the 60-dimensional pair feature vector for a single source-target pair.

    Pair representation: [x_source, x_target, |x_source-x_target|, x_source*x_target]

    Args:
        src_features: Source node features, shape (15,).
        tgt_features: Target node features, shape (15,).

    Returns:
        Pair features, shape (60,).

    Raises:
        FeatureContractError: If input dimensions are wrong.
    """
    src = validate_node_features(src_features, context="Source features")
    tgt = validate_node_features(tgt_features, context="Target features")

    abs_diff = np.abs(src - tgt)
    product = src * tgt

    pair = np.concatenate([src, tgt, abs_diff, product])

    if pair.shape[0] != PAIR_FEATURE_COUNT:
        raise FeatureContractError(
            f"Pair features: Expected {PAIR_FEATURE_COUNT}, got {pair.shape[0]}"
        )

    return pair.astype(np.float64)


def build_pair_features_batch(
    src_features: np.ndarray,
    tgt_features: np.ndarray,
    *,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Build pair features for multiple source-target pairs in bounded chunks.

    Args:
        src_features: Source features, shape (n, 15).
        tgt_features: Target features, shape (n, 15).
        chunk_size: Number of pairs per chunk.

    Returns:
        Pair features, shape (n, 60).

    Raises:
        FeatureContractError: If input dimensions are wrong.
    """
    src = validate_node_features(src_features, context="Batch source features")
    tgt = validate_node_features(tgt_features, context="Batch target features")

    if src.shape[0] != tgt.shape[0]:
        raise FeatureContractError(
            f"Source and target must have same length, got {src.shape[0]} vs {tgt.shape[0]}"
        )

    n = src.shape[0]
    result = np.empty((n, PAIR_FEATURE_COUNT), dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        s = src[start:end]
        t = tgt[start:end]
        result[start:end] = np.concatenate(
            [s, t, np.abs(s - t), s * t], axis=1
        )

    return result


def lookup_neuron_features(
    root_id: int,
    id_to_idx: Dict[int, int],
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Look up a neuron's 15-dimensional feature vector by root_id.

    Args:
        root_id: FlyWire neuron ID.
        id_to_idx: Mapping from root_id to feature matrix index.
        feature_matrix: Loaded feature matrix, shape (n, 15).

    Returns:
        Feature vector, shape (15,).

    Raises:
        InvalidNeuronIDError: If root_id not found.
    """
    if root_id not in id_to_idx:
        raise InvalidNeuronIDError(
            f"Neuron ID {root_id} not found in the FlyMind dataset."
        )
    idx = id_to_idx[root_id]
    return feature_matrix[idx]
