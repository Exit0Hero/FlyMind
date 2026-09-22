#!/usr/bin/env python3
"""Live HTTP smoke test for the FlyMind API (Phase 16).

Brings the real FastAPI app up and probes the exact HTTP contract that compose,
CI and pre-rollout checks depend on:

    /api/health  -> always 200 process-wide; `status` flips ok/degraded
    /api/ready   -> 200 iff the model artifact is loaded; 503 otherwise
    /api/version -> 200, metadata-only, MUST NOT force a model load

The app is run WITHOUT the model artifact (none is needed): readiness is
expected to be 503 ("not ready") because nothing is attached — asserting that
503 is what proves the readiness gate works before an operator ever mounts a
real model. No dataset is touched either, so this can run on any clean runner
(CI) and on a fresh checkout.

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
        health = client.get("/api/health")
        assert health.status_code == 200, f"health={health.status_code}: {health.text}"
        body = health.json()
        assert body["status"] == "degraded", "model should not be loaded"
        assert body["model_loaded"] is False

        ready = client.get("/api/ready")
        assert ready.status_code == 503, f"ready should be 503 without artifact, got {ready.status_code}"

        version = client.get("/api/version")
        assert version.status_code == 200, f"version={version.status_code}: {version.text}"
        data = version.json()
        for k in EXPECTED:
            assert k in data, f"missing {k!r} in {data}"
        assert data["model_loaded"] is False, "version probe must never load the model"


def main() -> int:
    probe()
    print("smoke ok: health=degraded, ready=503(unloaded), version=200(no model load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())