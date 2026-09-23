"""Model artifact safety guard.

Responsible for verifying that the RF artifact we serve is the artifact the
research pipeline produced. Checks, in order:

1. Artifact existence (model pickle + metadata).
2. SHA-256 integrity of the pickle against the value recorded in metadata.
3. Structural contract after load: model type, feature dimensionality, and
   the 15 node-feature -> 60 pair-feature shape.

None of these checks modify the model, retrain anything, or alter prediction
output; they only fail fast when the served artifact violates the contract.

All validation failures surface as :class:`app.core.errors.ModelUnavailableError`
with client-safe messages; full details are logged server-side.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import ModelUnavailableError

log = logging.getLogger("flymind.model_guard")

EXPECTED_NODE_FEATURES = 15
EXPECTED_PAIR_FEATURES = 60

_MODEL_CLASS_HINT = "RandomForestClassifier"

_DEFAULT_METADATA_NODE_COUNT = 15
_DEFAULT_METADATA_PAIR_COUNT = 60


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata JSON defensively; returns ``{}`` when unreadable."""
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError("metadata is not an object")
        return data
    except (OSError, json.JSONDecodeError, TypeError):
        log.warning("metadata unreadable - continuing without it", extra={"fields": {"path": str(metadata_path)}})
        return {}


def verify_artifact_integrity(
    *,
    model_path: Path,
    metadata_path: Path,
    integrity_check_enabled: bool = True,
) -> dict[str, Any]:
    """Verify that the model artifact exists and matches its recorded hash.

    Raises :class:`ModelUnavailableError` when a fatal violation is detected.
    Returns a summary dict for logging.
    """
    summary: dict[str, Any] = {"model_path": str(model_path), "ok": True, "notes": []}

    if not model_path.exists():
        raise ModelUnavailableError("Model artifact is not available on this instance")

    metadata = read_metadata(metadata_path)
    summary["metadata_present"] = bool(metadata)

    expected_hash = metadata.get("artifact_hash_sha256")

    if integrity_check_enabled and expected_hash:
        actual = sha256_file(model_path)
        if actual != str(expected_hash).lower():
            log.error(
                "model checksum mismatch",
                extra={"fields": {"expected": expected_hash, "actual": actual}},
            )
            raise ModelUnavailableError(
                "Model artifact integrity check failed; refusing to serve"
            )
        summary["checksum_verified"] = True
        summary["checksum_match"] = True
    else:
        summary["checksum_verified"] = False
        if not integrity_check_enabled:
            summary["notes"].append("integrity check disabled by configuration")
        else:
            summary["notes"].append("no recorded checksum in metadata; skipped")

    log.info("model artifact verified", extra={"fields": summary})
    return summary


def validate_loaded_model(
    inference: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structural validation of a loaded FlyMindInference instance.

    Detects a wrong model class, mismatched feature dimensions, or a broken
    feature contract. Normalizes/validates only; never re-trains or edits.

    Raises :class:`ModelUnavailableError` on violation.
    """
    metadata = metadata or {}
    expected_node = metadata.get("node_feature_count") or _DEFAULT_METADATA_NODE_COUNT
    expected_pair = metadata.get("pair_feature_count") or _DEFAULT_METADATA_PAIR_COUNT
    canonical = metadata.get("canonical_features")

    report: dict[str, Any] = {
        "ok": True,
        "node_feature_count": expected_node,
        "pair_feature_count": expected_pair,
        "notes": [],
    }

    writer = inference._rf if hasattr(inference, "_rf") else None
    if writer is None:
        raise ModelUnavailableError("Model object was not loaded")

    model_class = type(writer).__name__
    if _MODEL_CLASS_HINT not in model_class:
        raise ModelUnavailableError(
            f"Model artifact is not a RandomForest (got {model_class})"
        )
    report["model_class"] = model_class

    feature_cols = getattr(inference, "_feature_cols", None)
    if not feature_cols:
        raise ModelUnavailableError("Loaded model has no feature column metadata")
    node_feature_count = len(feature_cols)
    if node_feature_count != expected_node:
        raise ModelUnavailableError(
            f"Feature contract mismatch: expected {expected_node} node features, got {node_feature_count}"
        )
    report["node_features_loaded"] = node_feature_count

    x_matrix = getattr(inference, "_X", None)
    if x_matrix is not None and x_matrix.shape[1] != expected_node:
        raise ModelUnavailableError(
            f"Feature matrix shape mismatch: expected {expected_node} columns, got {x_matrix.shape[1]}"
        )

    try:
        pair_dim = int(inference.feature_dim)
    except (TypeError, ValueError, AttributeError):
        pair_dim = None
    if pair_dim is not None and pair_dim != expected_pair:
        raise ModelUnavailableError(
            f"Pair feature dimensionality mismatch: expected {expected_pair}, got {pair_dim}"
        )
    report["pair_features_loaded"] = pair_dim

    if canonical and list(feature_cols) != list(canonical):
        log.warning(
            "feature order differs from metadata canonical order",
            extra={"fields": {"loaded": list(feature_cols), "canonical": list(canonical)}},
        )
        report["notes"].append("feature order differs from canonical; ordering preserved from model")

    log.info("loaded model validated", extra={"fields": report})
    return report


def guard_model_load(inference: Any) -> dict[str, Any]:
    """Single entry point used by the ML service after a successful load."""
    metadata = read_metadata(settings.MODEL_METADATA_PATH)
    return validate_loaded_model(inference, metadata=metadata)