"""FlyMind deployment guide (Phase 16).

This document is the operational companion to the repository. It covers the
two container images (backend FastAPI + frontend Next.js), the compose stack
that runs them with zero infrastructure, the environment contract, health /
readiness semantics, and the safe rollout/rollback workflow — including how the
model artifact and datasets travel (and never, ever, get baked into an image
or committed to git).

Prereqs
-------
* Docker Engine >= 24 (with buildx; compose v2).
* FlyMind checked out: the repo is the build context. Model artifacts live on
  the HOST under `models/`; `data/` holds datasets. Neither enters the image.

-------------------------------------------------------
1. The environment contract
-------------------------------------------------------

Every runtime value is read from the environment and **must** be prefixed with
`FLYMIND_` (backend) — see `backend/app/core/config.py`. The frontend reads
`NEXT_PUBLIC_*` (browser bundle, baked at build time) and `BACKEND_URL` (server
process, runtime), see `frontend/next.config.ts` + `frontend/lib/api.ts`.

Copy `.env.example` to `.env` at the repo root; compose reads it. Anything
secret (API keys, a real `*.pkl` path, credentials) belongs ONLY in `.env`,
which is git-ignored and never copied into a build context.

Contract check (CI enforces this):
  * backend: every `FLYMIND_*`-named setting listed in `.env.example` must be
    present in the Settings model (config.py) — added/removed settings are
    caught before merge.
  * frontend: `frontend/lib/api.ts` must read `NEXT_PUBLIC_API_URL` and fall
    back to the same-origin `/api` proxy; `next.config.ts` proxies
    `/api/:path*` -> `BACKEND_URL` (rewritten, so the browser only ever knows
    one origin).

-------------------------------------------------------
2. Image design + the data/model strategy
-------------------------------------------------------

Both images are multi-stage and run as an unprivileged user. The model artifact
and datasets are intentionally **not** part of the build: you do not rebuild the
image to ship a new model, because a model swap is a *runtime* operation.

  * `models/`  (host)  -> mounted read-only into the backend container.
  * `data/`    (host)  -> mounted read-only into the backend container.
  * `results/` (host)  -> mounted read-write (the API writes predictions).

The 177 MB `link_prediction_rf.pkl` lives on the host and is volume-mounted;
the layered image therefore stays small and a model refresh is `mv model.pkl &&`
restart, not a rebuild. Because the image never contains the artifact, it is
safe to publish, cache and pull anywhere (registry/CI) with no data-exposure.

Readiness contract (two probes — don't confuse them):
  * `/api/health` — Liveness. 200 always once the process is up; `status` is
    `ok` when the model is loaded, `degraded` otherwise. Used by compose as the
    service gate so the stack can come up before an artifact is mounted.
  * `/api/ready`  — Readiness. **200 only when the model artifact is loaded and
    verified**; 503 otherwise. This is the probe orchestration (and CI rollout
    gates) must use for *traffic* decisions, because FlyMind cannot serve
    predictions until the .pkl is loaded and its metadata matches.
  * `/api/version` visited by CI/preflight expects a 200 with
    `{app_version, api_version, model_version, model_loaded}` — and must never
    trigger a model load.

See `docs/DEPLOYMENT.md` (this file) app diagram below for how the two probes
sit on each container.

```
                              ┌──────────────────────────────────────────────┐
                              │                    compose                    │
    browser ──:3000─────────► │  frontend (Next.js, non-root, HEALTHCHECK     │
                              │   probes /api/health via proxy to backend)    │
                              │      │ rewrites /api/*                        │
                              │      ▼                                        │
                              │  backend (FastAPI, non-root, HEALTHCHECK       │
                              │   probes /api/health)   ▲ mounts:             │
                              │   read-only: models/ data/                    │
                              │   read-write: results/                        │
                              └───────────────┴──────────────────────────────┘
   model/ dataset live on the host machine entirely (modvol mount);
   no database / cache / queue / broker infrastructure is created or managed.
```

-------------------------------------------------------
3. Build & run
-------------------------------------------------------

    # 3.1 compose stack (recommended)
    cp .env.example .env
    docker compose build
    docker compose up -d
    docker compose ps          # both should show "healthy" once artifact present
    open http://localhost:3000

    # 3.2 just the backend (no frontend)
    docker run --rm -p 8001:8001 \
        -v "$PWD/models:/flymind/models:ro" \
        -v "$PWD/data:/flymind/data:ro" \
        --env-file .env flymind-backend:local

    # 3.3 just the frontend (no backend — frontend will report degraded via API)
    docker run --rm -p 3000:3000 \
        -e BACKEND_URL=http://localhost:8001 \
        --env-file .env flymind-frontend:local

    # 3.4 rebuild after changes / flush containers
    docker compose build --pull && docker compose up -d --force-recreate

Frontend + backend talk on the compose network: frontend calls the backend at
`http://backend:8001` (service name, resolved by compose DNS) — not localhost.
The browser never talks directly to the backend: `/api/*` is proxied, keeping a
single origin and clean CORS.

-------------------------------------------------------
4. Environment variables
-------------------------------------------------------

Requires (must be set; compose refuses to start otherwise):
  * `FLYMIND_CORS_ORIGINS` — list of allowed browser origins for /api (frontend
    same-origin proxy makes this narrow; keep it tight).

Recommended / tuned for deployment:
  * `FLYMIND_ENV=prod`, `FLYMIND_DEBUG=false` (dev only; no reload).
  * `FLYMIND_LOG_LEVEL=info` (default), raise to `warning` in high-volume PROD.
  * `FLYMIND_HOST`, `FLYMIND_PORT` — listener address (default 0.0.0.0:8001).
  * `FLYMIND_MODEL_FILENAME` / `FLYMIND_MODEL_METADATA_FILENAME` — artifact
    names under `MODELS_DIR` (defaults point at `link_prediction_rf(.pkl|.metadata.json)`).
  * `NEXT_PUBLIC_API_URL` — where the browser sends /api calls. Default `/api`
    (same-origin proxy). Set to a full URL only for a truly separate API host.
  * `BACKEND_URL` — the *server-side* upstream the Next rewrite targets when the
    browser hits same-origin `/api/*`. In compose: `http://backend:8001`.
    On the split production path (Vercel + backend on this machine) set it in
    the Vercel project (Production) to a publicly reachable URL for port 8001.
    This host has no inbound port forwarding (CGNAT) and a dynamic public IP,
    so production uses a Cloudflare quick tunnel
    (`https://*.trycloudflare.com` → `http://127.0.0.1:8001`); restarts of the
    tunnel yield a new hostname and require re-setting `BACKEND_URL` + redeploy.

Frontend auto-guess on cold start: if the frontend starts with no backend
reachable it keeps serving; the Dashboard surfaces a "degraded" banner (via
`/api/health`) and the rewrite is still configured — it only 502s on the
proxied paths until the backend is healthy.

-------------------------------------------------------
5. Health checks / monitoring
-------------------------------------------------------

* Container liveness/readiness comes from `docker HEALTHCHECK`s wired to
  `_api/health` (see the Dockerfiles). `docker ps`/compose `ps` will show the
  health column.
* Cross-service ordering in compose uses `depends_on: backend: condition:
  service_healthy` so the frontend only starts after the backend passes its
  liveness probe (not engine boot), short-circuiting the classic
  "frontend up before backend" race while still allowing bring-up before any
  model is attached.
* For real traffic gating (LB/k8s/CI rollouts) use `/api/ready`; expecting
  `503` until the artifact is loaded is CORRECT behaviour — most "the container
  fails readiness" reports in FlyMind are actually "the model isn't attached
  yet", and `/api/version.model_loaded` tells you which case you're in without
  guessing.

-------------------------------------------------------
6. CI (app-ci.yml)
-------------------------------------------------------

The `app-ci` workflow runs on PR + push to main and is fully isolated from your
data and model:
  * backend — pip-installs `requirements-dev.txt` (pytest/httpx/ruff added on
    top of runtime deps), `ruff check`, then the FULL suite with the stub ML
    service (no dataset, no artifact, no network) plus a live HTTP smoke probe
    of /api/health, /api/ready and /api/version contracts.
  * frontend — `npm ci`, `tsc --noEmit`, `next build`.
  * All set to keep runtimes green without ever touching experiments or models/.

The legacy Streamlit-era `ci.yml` has been removed (it could not pass on a
clean runner: missing pytest install, Docker job required git-ignored data).
`app-ci.yml` is the single CI gate: backend lint/tests, frontend lint/typecheck/build,
API smoke, and docker build of both images.

-------------------------------------------------------
7. Troubleshooting
-------------------------------------------------------

  symptom: frontend healthy but /api returns 502
     → check `BACKEND_URL` in the frontend env resolves to the compose service
       name (`http://backend:8001`), not `localhost`, and the backend container
       is up: `docker compose logs backend`.

  symptom: backend HEALTHCHECK shows unhealthy / /api/ready=503
     → the model is not attached or not loaded. Confirm the bind mount:
       `docker inspect <backend> --format '{{json .Mounts}}'` and that
       models/link_prediction_rf.pkl exists on the host. `/api/version` shows
       `model_loaded:false` to confirm this is the cause.

  symptom: compose complains about FLYMIND_CORS_ORIGINS
     → ensure `.env` was copied from `.env.example` (compose refuses to start
       without it). Do not set a `*` value in production.

  symptom: image "huge"
     → it is multi-stage and lean; the big .pkl is never in an image. If it
       still looks big, the frontend runs `next build` of node_modules in an
       intermediate stage — the runtime layer copies only compiled output and
       production deps. Run `docker images flymind-*` to diff layer sizes.

  rollback
     → images are tagged `local` by default. Rollback = start an older tag:
       `docker compose run --rm ... <old-tag>` or simply
       `git checkout <release> && docker compose up -d --build`. Since the
       model is a volume mount (not a layer), rolling back code is independent
       of rolling the model.

-------------------------------------------------------
8. Security notes
-------------------------------------------------------

  * Runner: non-root `flymind` user in both images (no root shell, minimal
    surface).
  * Secrets: only via env (compose env_file + `.env`), never in files inside
    the image. `.env`, models, data and experiments are build-context-excluded
    via `.dockerignore` files in each subdir.
  * No infra credentials: this stack creates/manages no DB/cache/queue/broker;
    there is no secret store to configure.
  * Dependencies: `npm ci` (lockfile-pinned) for the frontend and pinned
    floors (`>=`) in backend `requirements*.txt`; CI + CI installs the same
    files so the image and the test env match.
  * Health/ready intentionally degrade but never panic: the API answers 503
    with a reason — not a crash — so orchestrators can read the contract.
"""
