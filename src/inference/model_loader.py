"""FlyMind Production Inference — Safe model loading with validation."""

import hashlib
import logging
import pathlib
import pickle
import time
from typing import Any, Dict, Optional

from .exceptions import ModelLoadError, ModelArtifactMismatchError
from .model_contract import (
    MODEL_TYPE, EXPECTED_ESTIMATORS, NODE_FEATURE_COUNT,
    PAIR_FEATURE_COUNT, CANONICAL_NODE_FEATURES, EXPECTED_ARTIFACT_HASH,
)

logger = logging.getLogger(__name__)


def _compute_hash(filepath: pathlib.Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_artifact(
    model_path: pathlib.Path,
    *,
    verify_hash: bool = True,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate the model pickle artifact.

    Returns:
        Dict with keys 'model', 'feature_cols'.

    Raises:
        ModelLoadError: If file is missing or unreadable.
        ModelArtifactMismatchError: If contents do not match contract.
    """
    if expected_hash is None:
        expected_hash = EXPECTED_ARTIFACT_HASH

    t0 = time.perf_counter()

    # Check file exists
    if not model_path.exists():
        raise ModelLoadError(f"Model artifact not found: {model_path}")

    if not model_path.is_file():
        raise ModelLoadError(f"Model path is not a file: {model_path}")

    # Compute hash
    if verify_hash:
        actual_hash = _compute_hash(model_path)
        if actual_hash != expected_hash:
            raise ModelArtifactMismatchError(
                f"Artifact hash mismatch. Expected {expected_hash}, got {actual_hash}. "
                "This model file may have been modified or corrupted."
            )
        logger.info("Model artifact hash verified: %s", actual_hash)

    # Load pickle
    try:
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
    except Exception as e:
        raise ModelLoadError(f"Failed to deserialize model: {e}") from e

    # Validate structure
    if not isinstance(artifact, dict):
        raise ModelArtifactMismatchError(
            f"Expected dict artifact, got {type(artifact).__name__}"
        )

    required_keys = {"model", "feature_cols"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ModelArtifactMismatchError(
            f"Missing required keys in artifact: {missing}"
        )

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Validate model type
    actual_type = type(model).__name__
    if actual_type != MODEL_TYPE:
        raise ModelArtifactMismatchError(
            f"Expected model type {MODEL_TYPE}, got {actual_type}"
        )

    # Validate estimators
    if hasattr(model, "n_estimators"):
        if model.n_estimators != EXPECTED_ESTIMATORS:
            raise ModelArtifactMismatchError(
                f"Expected {EXPECTED_ESTIMATORS} estimators, got {model.n_estimators}"
            )

    # Validate feature dimension
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != PAIR_FEATURE_COUNT:
            raise ModelArtifactMismatchError(
                f"Expected {PAIR_FEATURE_COUNT} features, got {model.n_features_in_}"
            )

    # Validate feature cols count
    if len(feature_cols) != NODE_FEATURE_COUNT:
        raise ModelArtifactMismatchError(
            f"Expected {NODE_FEATURE_COUNT} feature names, got {len(feature_cols)}"
        )

    # Validate feature order matches canonical
    if list(feature_cols) != list(CANONICAL_NODE_FEATURES):
        raise ModelArtifactMismatchError(
            f"Feature order mismatch. Expected {CANONICAL_NODE_FEATURES}, got {list(feature_cols)}"
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Model loaded and validated in %.3fs: type=%s, estimators=%d, features=%d",
        elapsed, actual_type, model.n_estimators, model.n_features_in_,
    )

    return artifact
