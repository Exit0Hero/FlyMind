"""ML service wrapping FlyMindInference with lazy loading and safety guards.

Lazy-loading is thread-safe (double-checked locking) and runs the model
integrity + feature-contract guard before the artifact is served. Any load or
validation failure is logged in full server-side but exposed to clients only
as a :class:`app.core.errors.ModelUnavailableError` (503).
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from typing import Any

from app.core.config import settings
from app.core.errors import ModelUnavailableError
from app.services.model_guard import (
    guard_model_load,
    read_metadata,
    verify_artifact_integrity,
)

log = logging.getLogger("flymind.ml_service")


class MLService:
    """Thread-safe lazy-loading wrapper around FlyMindInference."""

    def __init__(self) -> None:
        self._inference: Any = None
        self._datastore: Any = None
        self._lock = threading.Lock()
        self._loaded = False
        self._error: str | None = None

    def _ensure_src_on_path(self) -> None:
        project_root = str(settings.PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def _load(self) -> None:
        if self._loaded or self._error is not None:
            return
        with self._lock:
            if self._loaded or self._error is not None:
                return
            try:
                self._ensure_src_on_path()

                # 1. Artifact integrity (exists? sha-256 matches metadata?)
                verify_artifact_integrity(
                    model_path=settings.MODEL_PATH,
                    metadata_path=settings.MODEL_METADATA_PATH,
                    integrity_check_enabled=settings.ENABLE_MODEL_INTEGRITY_CHECK,
                )

                # 2. Load the research inference layer.
                from src.link_prediction.inference import FlyMindInference

                inference = FlyMindInference()
                inference._ensure_loaded()

                # 3. Structural / feature-contract validation.
                guard_model_load(inference)

                self._inference = inference
                self._loaded = True
                log.info(
                    "model loaded",
                    extra={
                        "fields": {
                            "version": settings.MODEL_VERSION,
                            "path": settings.MODEL_PATH.name,
                            "neurons": inference.n_neurons,
                            "edges": inference.n_edges,
                        }
                    },
                )
            except ModelUnavailableError as exc:
                log.error("model load rejected", exc_info=exc)
                self._error = exc.message
            except Exception:
                log.exception("model failed to load")
                self._error = "ML model could not be loaded on this instance"

    def _ensure_loaded(self) -> None:
        self._load()
        if self._error is not None:
            raise ModelUnavailableError(self._error)
        if not self._loaded:
            raise ModelUnavailableError("Model is not loaded yet")

    def ensure_load(self) -> bool:
        """Idempotent warm-up used by the readiness probe.

        Never raises. Returns True when the model is loaded afterwards.
        Failures are recorded on :attr:`load_error` so `/api/ready` can
        surface a client-safe reason instead of hanging on a cold start.
        """
        self._load()
        return self._loaded

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._error

    def get_neuron(self, root_id: int) -> Any:
        self._ensure_loaded()
        return self._inference.get_neuron(root_id)

    def score_connection(self, source_root_id: int, target_root_id: int) -> Any:
        self._ensure_loaded()
        return self._inference.score_connection(source_root_id, target_root_id)

    def rank_candidate_targets(
        self,
        source_root_id: int,
        k: int = 10,
        candidate_pool_size: int = 1000,
    ) -> list[Any]:
        self._ensure_loaded()
        # Defense in depth: clamp to configured caps even if a caller bypasses
        # the API validation layer.
        k = max(1, min(k, settings.MAX_K))
        candidate_pool_size = max(1, min(candidate_pool_size, settings.MAX_CANDIDATES))
        return self._inference.rank_candidate_targets(
            source_root_id, k=k, candidate_pool_size=candidate_pool_size,
        )

    @property
    def n_neurons(self) -> int:
        self._ensure_loaded()
        return self._inference.n_neurons

    @property
    def n_edges(self) -> int:
        self._ensure_loaded()
        return self._inference.n_edges

    @property
    def feature_dim(self) -> int:
        self._ensure_loaded()
        return self._inference.feature_dim

    @property
    def node_feature_dim(self) -> int:
        self._ensure_loaded()
        return self._inference.node_feature_dim

    @property
    def feature_names(self) -> list[str]:
        self._ensure_loaded()
        return list(self._inference._feature_cols)

    @property
    def model_version(self) -> str:
        # Prefer the frozen artifact metadata so the version surface tracks a
        # model swap without a code deploy; fall back to settings.
        meta = read_metadata(settings.MODEL_METADATA_PATH)
        return str(meta.get("model_version") or settings.MODEL_VERSION)

    @property
    def model_artifact(self) -> str:
        meta = read_metadata(settings.MODEL_METADATA_PATH)
        return str(meta.get("artifact_filename") or settings.MODEL_FILENAME)

    @property
    def model_info(self) -> dict[str, Any]:
        self._ensure_loaded()
        rf = self._inference._rf
        return {
            "n_estimators": getattr(rf, "n_estimators", None),
            "n_classes": len(getattr(rf, "classes_", [])),
            "class_labels": [int(c) for c in getattr(rf, "classes_", [])],
        }

    def search_neurons(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search neurons by root_id (exact/prefix), name (substring), or primary_type (substring)."""
        self._ensure_loaded()
        limit = min(max(int(limit), 1), settings.MAX_SEARCH_LIMIT)
        lookup = self._inference._nt_lookup
        results: list[dict[str, Any]] = []

        query_str = str(query).strip()
        query_lower = query_str.lower()
        query_int: int | None = None
        try:
            query_int = int(query_str)
        except (ValueError, TypeError):
            pass

        def _str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, float) and math.isnan(val):
                return ""
            return str(val)

        # Exact root_id match first
        if query_int is not None and query_int in lookup:
            info = lookup[query_int]
            results.append({
                "root_id": query_int,
                "name": _str(info.get("name")),
                "nt_type": _str(info.get("nt_type")),
                "super_class": _str(info.get("super_class")),
                "primary_type": _str(info.get("primary_type")),
            })

        # Prefix / substring search on root_id, name, and primary_type
        if len(results) < limit:
            for rid, info in lookup.items():
                if len(results) >= limit:
                    break
                if results and any(r["root_id"] == rid for r in results):
                    continue
                rid_str = str(rid)
                name = _str(info.get("name")).lower()
                primary_type = _str(info.get("primary_type")).lower()
                if (
                    query_str in rid_str
                    or query_lower in name
                    or query_lower in primary_type
                ):
                    results.append({
                        "root_id": rid,
                        "name": _str(info.get("name")),
                        "nt_type": _str(info.get("nt_type")),
                        "super_class": _str(info.get("super_class")),
                        "primary_type": _str(info.get("primary_type")),
                    })

        return results[:limit]

    def get_model_obj(self) -> Any:
        """Return the underlying RF model object."""
        self._ensure_loaded()
        return self._inference._rf


ml_service = MLService()