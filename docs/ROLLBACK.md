# FlyMind Rollback Strategy

## Current Release

- **Application:** FlyMind v1.0.0
- **Model:** FlyMind-RF v1.0.0
- **Branch:** `feature/flymind-deployment`

## How to Rollback

### If using Docker

1. Stop the current container:
   ```bash
   docker stop flymind
   ```

2. Run the previous working image:
   ```bash
   docker run -d --name flymind -p 8501:8501 flymind:<previous-tag>
   ```

### If using Git

1. Checkout the last working commit:
   ```bash
   git checkout <commit-hash>
   ```

2. Rebuild and redeploy.

## Identifying a Broken Release

- Dashboard shows traceback or artifact error
- Health check returns `unhealthy`
- Predictions return unexpected scores
- Candidate ranking fails

## Model Artifact

The model artifact (`models/link_prediction_rf.pkl`) is frozen. If corrupted:

1. Restore from Git: `git checkout HEAD -- models/link_prediction_rf.pkl`
2. Verify hash: `sha256sum models/link_prediction_rf.pkl`
3. Expected: `c6cdf6ca49d9216b8a7eff8e2b90cf9c0bfa8f0619fc1b5bfed74c7b65180316`

## Version History

| Version | Commit | Date | Notes |
|---------|--------|------|-------|
| v1.0.0 | 9485b16 | 2026-09-20 | Production inference hardened |
| v1.0.0 | 3a9c4cc | 2026-09-20 | Public release packaging |
