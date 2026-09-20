"""FlyMind Production Inference — Health check and startup validation."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .model_contract import (
    MODEL_VERSION, EXPECTED_ESTIMATORS, NODE_FEATURE_COUNT,
    PAIR_FEATURE_COUNT, CANONICAL_NODE_FEATURES, EXPECTED_ARTIFACT_HASH,
)
from .model_loader import load_model_artifact

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check result."""
    status: str  # "healthy" or "unhealthy"
    model_version: str
    feature_dimension: int
    artifact_verified: bool
    details: str = ""
    duration_ms: float = 0.0


def check_model_health(
    models_dir: Path,
    lp_dir: Path,
    *,
    verify_hash: bool = True,
) -> HealthStatus:
    """Verify model artifact and processed data are valid.

    Args:
        models_dir: Path to models directory.
        lp_dir: Path to link_prediction data directory.
        verify_hash: Whether to verify artifact hash.

    Returns:
        HealthStatus with detailed results.
    """
    t0 = time.perf_counter()
    details = []

    try:
        # Check model artifact
        model_path = models_dir / "link_prediction_rf.pkl"
        artifact = load_model_artifact(model_path, verify_hash=verify_hash)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        artifact_verified = True
        details.append("Model artifact: OK")

        # Check feature dimension
        if model.n_features_in_ != PAIR_FEATURE_COUNT:
            details.append(f"Feature dimension mismatch: {model.n_features_in_} != {PAIR_FEATURE_COUNT}")
            return HealthStatus(
                status="unhealthy",
                model_version=MODEL_VERSION,
                feature_dimension=model.n_features_in_,
                artifact_verified=False,
                details="; ".join(details),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        details.append("Feature dimension: OK")

        # Check estimator count
        if model.n_estimators != EXPECTED_ESTIMATORS:
            details.append(f"Estimator count mismatch: {model.n_estimators} != {EXPECTED_ESTIMATORS}")
            return HealthStatus(
                status="unhealthy",
                model_version=MODEL_VERSION,
                feature_dimension=model.n_features_in_,
                artifact_verified=artifact_verified,
                details="; ".join(details),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        details.append("Estimator count: OK")

        # Check feature names
        if list(feature_cols) != list(CANONICAL_NODE_FEATURES):
            details.append("Feature names mismatch")
            return HealthStatus(
                status="unhealthy",
                model_version=MODEL_VERSION,
                feature_dimension=model.n_features_in_,
                artifact_verified=artifact_verified,
                details="; ".join(details),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        details.append("Feature names: OK")

        # Check required processed files
        required_files = [
            lp_dir / "X_features.npy",
            lp_dir / "id_to_idx.json",
            lp_dir / "edges_aggregated.parquet",
        ]
        for f in required_files:
            if not f.exists():
                details.append(f"Missing: {f.name}")
                return HealthStatus(
                    status="unhealthy",
                    model_version=MODEL_VERSION,
                    feature_dimension=PAIR_FEATURE_COUNT,
                    artifact_verified=artifact_verified,
                    details="; ".join(details),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
        details.append("Processed data files: OK")

        # Quick smoke test
        test_feats = np.zeros((1, PAIR_FEATURE_COUNT), dtype=np.float64)
        try:
            model.predict_proba(test_feats)
            details.append("Smoke test: OK")
        except Exception as e:
            details.append(f"Smoke test failed: {e}")
            return HealthStatus(
                status="unhealthy",
                model_version=MODEL_VERSION,
                feature_dimension=PAIR_FEATURE_COUNT,
                artifact_verified=artifact_verified,
                details="; ".join(details),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        return HealthStatus(
            status="healthy",
            model_version=MODEL_VERSION,
            feature_dimension=PAIR_FEATURE_COUNT,
            artifact_verified=artifact_verified,
            details="; ".join(details),
            duration_ms=elapsed,
        )

    except Exception as e:
        details.append(f"Error: {e}")
        elapsed = (time.perf_counter() - t0) * 1000
        return HealthStatus(
            status="unhealthy",
            model_version=MODEL_VERSION,
            feature_dimension=0,
            artifact_verified=False,
            details="; ".join(details),
            duration_ms=elapsed,
        )
