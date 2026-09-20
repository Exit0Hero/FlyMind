# FlyMind Deployment Platform Decision

## Options Evaluated

| Platform | RAM | Free Tier | Container | Public URL | Notes |
|----------|-----|-----------|-----------|------------|-------|
| Streamlit Community Cloud | 1 GB | Yes | No (Git-based) | Yes | Simplest, but 1 GB limit |
| Render | 512 MB free | Yes | Yes | Yes | Good for containers |
| Railway | 512 MB free | Yes | Yes | Yes | Simple deployment |
| Fly.io | 256 MB free | Yes | Yes | Yes | Low free tier |
| Local Docker | Unlimited | N/A | Yes | No | Development only |

## Decision: Streamlit Community Cloud

**Primary choice:** Streamlit Community Cloud

**Why:**
1. Native Streamlit support (no Docker needed)
2. Free public URL
3. Simple Git-based deployment
4. 1 GB RAM sufficient for FlyMind (~300 MB needed)
5. No infrastructure management

**Fallback:** Render (if SCC limitations are hit)

## Limitations of Streamlit Community Cloud

- 1 GB RAM limit (FlyMind needs ~300 MB — sufficient)
- No persistent storage (model baked into repo)
- Git-based deployment (large repo may be slow)
- No custom domains on free tier

## Deployment Steps

1. Push to GitHub repository
2. Connect to Streamlit Community Cloud
3. Set main file: `app/streamlit_app.py`
4. Deploy

## Public URL

To be determined after deployment.
