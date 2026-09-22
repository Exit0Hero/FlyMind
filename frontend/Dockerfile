# FlyMind frontend production image.
#
# Strategy
#   * Multi-stage: deps -> build -> runtime keeps the final layer small.
#   * Deterministic install via `npm ci` (package-lock.json is committed).
#   * Non-root runtime user; no dev tools (typescript/tailwind) in the image.
#   * The Next.js server is the WSGI-less `next start` process; it proxies
#     /api/* to the backend through rewrites using BACKEND_URL (runtime env).
#   * Healthcheck routes through the same rewrite chain (/api/health) so it
#     verifies the proxy hop to the backend, not just that Node is alive.

FROM node:22-bookworm-slim AS deps

WORKDIR /app

# package.json + lockfile first (cache-friendly; layer only busts on dep change).
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

# ---
FROM deps AS build

COPY . .

ARG NEXT_PUBLIC_API_URL=/api
ARG BACKEND_URL=http://backend:8001

ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    BACKEND_URL=$BACKEND_URL

RUN npm run build

# ---
FROM node:22-bookworm-slim AS runtime

# NODE_ENV=production makes node_modules slim (no dev deps stay resident).
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    NEXT_PUBLIC_API_URL=/api \
    BACKEND_URL=http://backend:8001

RUN groupadd --system flymind && useradd --system --gid flymind --home-dir /app flymind

WORKDIR /app

# Runtime layer needs only the compiled build + production deps.
COPY --from=deps --chown=flymind:flymind /app/node_modules ./node_modules
COPY --from=build --chown=flymind:flymind /app/.next ./.next
COPY --from=build --chown=flymind:flymind /app/package.json ./package.json

USER flymind

# The ready gate runs from inside the container; it shares the container
# network, so 127.0.0.1 is the process itself and the /api rewrite goes to the
# backend.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"]

EXPOSE 3000

CMD ["node_modules/.bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]
