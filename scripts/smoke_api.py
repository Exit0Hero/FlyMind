#!/usr/bin/env python3
"""Live HTTP smoke test for the FlyMind API (Phase 16/17).

Brings the real FastAPI app up and probes the exact HTTP contract that compose,
CI and pre-rollout checks depend on:

    /api/health  -> always 200 process-wide; `status` flips ok/degraded
    /api/version -> 200, metadata-only, MUST NOT force a model load
    /api/ready   -> warms the model; 200 when loaded+verified, 503 otherwise

Order matters: health and version are probed BEFORE ready so their
`model_loaded:false` assertions prove neither endpoint triggers a load. ready
is last because it is the only probe allowed to warm the artifact.

On a clean CI runner (no models/ mount) ready is expected to be 503. On a
dev machine with the artifact present, ready may legitimately be 200 after a
successful warm-up — both are accepted as long as the contract above holds.

Exit code 0 == contract green.

Run from anywhere (backend/ is pinned ahead of sys.path below):
    python scripts/smoke_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
if "app" in sys.modules:
    sys.modules.pop("app")

from app.main import app
from starlette.testclient import TestClient

EXPECTED = ["app_version", "api_version", "model_version", "model_loaded"]


def probe() -> None:
    with TestClient(app) as client:
        # 1) Liveness — must never load the model.
        health = client.get("/api/health")
        assert health.status_code == 200, f"health={health.status_code}: {health.text}"
        body = health.json()
        assert body["status"] == "degraded", "model should not be loaded yet"
        assert body["model_loaded"] is False

        # 2) Version — metadata-only, still no load.
        version = client.get("/api/version")
        assert version.status_code == 200, f"version={version.status_code}: {version.text}"
        data = version.json()
        for k in EXPECTED:
            assert k in data, f"missing {k!r} in {data}"
        assert data["model_loaded"] is False, "version probe must never load the model"
        assert data["model_version"], "model_version must be known without a load"

        # 3) Readiness — the only probe allowed to warm the model.
        ready = client.get("/api/ready")
        assert ready.status_code in (200, 503), (
            f"ready should be 200 or 503, got {ready.status_code}"
        )
        if ready.status_code == 200:
            assert ready.json()["status"] == "ready"
        else:
            assert ready.json()["error"]["code"] == "NOT_READY"


def main() -> int:
    probe()
    print("smoke ok: health=degraded, version=200(no load), ready=200|503")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
