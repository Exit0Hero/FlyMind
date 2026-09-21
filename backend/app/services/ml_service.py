"""ML service wrapping FlyMindInference with lazy loading."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings


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
                from src.link_prediction.inference import FlyMindInference
                self._inference = FlyMindInference()
                self._loaded = True
            except Exception as exc:
                self._error = str(exc)

    def _ensure_loaded(self) -> None:
        self._load()
        if self._error:
            raise RuntimeError(f"ML model failed to load: {self._error}")

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
    def model_info(self) -> dict[str, Any]:
        self._ensure_loaded()
        rf = self._inference._rf
        return {
            "n_estimators": getattr(rf, "n_estimators", None),
            "n_classes": len(getattr(rf, "classes_", [])),
            "class_labels": [int(c) for c in getattr(rf, "classes_", [])],
        }

    def search_neurons(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search neurons by root_id (exact/prefix) or name (substring)."""
        self._ensure_loaded()
        lookup = self._inference._nt_lookup
        results: list[dict[str, Any]] = []

        query_str = str(query).strip()
        query_int: int | None = None
        try:
            query_int = int(query_str)
        except (ValueError, TypeError):
            pass

        # Exact root_id match first
        if query_int is not None and query_int in lookup:
            info = lookup[query_int]
            results.append({
                "root_id": query_int,
                "name": info.get("name"),
                "nt_type": info.get("nt_type"),
                "super_class": info.get("super_class"),
            })

        # Prefix / substring search on root_id and name
        if len(results) < limit:
            for rid, info in lookup.items():
                if len(results) >= limit:
                    break
                if results and any(r["root_id"] == rid for r in results):
                    continue
                rid_str = str(rid)
                name = info.get("name") or ""
                if (
                    query_str in rid_str
                    or query_str.lower() in name.lower()
                ):
                    results.append({
                        "root_id": rid,
                        "name": name,
                        "nt_type": info.get("nt_type"),
                        "super_class": info.get("super_class"),
                    })

        return results[:limit]

    def get_model_obj(self) -> Any:
        """Return the underlying RF model object."""
        self._ensure_loaded()
        return self._inference._rf


ml_service = MLService()
