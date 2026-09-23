# FlyMind Deployment Architecture

> **Status (Phase 17): SUPERSEDED.** The Streamlit container decision below is
> historical. The live architecture is the two-service stack in
> `docs/DEPLOYMENT.md` (FastAPI + Next.js) and, for production, the split
> Vercel frontend + host backend described there.

## Decision: Option A — Streamlit Container

```
Browser
   ↓
Streamlit (port 8501)
   ↓
Production Inference Layer
   ↓
Frozen RF Model (177 MB)
   ↓
Processed FlyMind Data (~40 MB)
```

## Why Option A

1. **Self-contained:** Single container, no external dependencies
2. **Simple:** No API gateway, no service mesh, no load balancer needed
3. **Sufficient:** Dashboard is the only user interface
4. **Low complexity:** One process, one port, one health check

## Why Not Option B (Streamlit + API)

- No operational benefit for single-user dashboard
- Adds deployment complexity
- Adds network latency
- Requires managing two services

## Why Not Option C (Other)

- No evidence of need for microservices
- No evidence of need for serverless
- No evidence of need for Kubernetes

## Expected Resources

| Resource | Expected |
|----------|----------|
| RAM | ~300 MB |
| CPU | 1 vCPU sufficient |
| Disk | ~500 MB (image + data) |
| Network | Single port 8501 |
| Startup | ~10-15 seconds |

## Security Implications

- Model artifact is trusted (version-controlled)
- No user-uploaded files
- No secrets in container
- Non-root user
- Single port exposure

## Limitations

- Single-instance only (no horizontal scaling)
- No persistent sessions
- No authentication (public dashboard)
- No HTTPS at application level (platform handles)
