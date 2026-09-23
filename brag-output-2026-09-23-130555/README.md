# FlyMind Launch Video

**Watch:** [brag.mp4](./brag.mp4) — 19s · 1920×1080 · 30fps · h264 + AAC

## Contents

| Path | Description |
|------|-------------|
| `brag.mp4` | Final rendered launch video |
| `brag-plan.md` | Creative brief, tone, storyboard |
| `composition-brief.md` | HyperFrames composition handoff |
| `composition/` | Source composition (index.html, assets, beats) |

## Research figures

Static result figures are in [`../results/figures/`](../results/figures/) and presentation slides in [`../results/presentation/`](../results/presentation/).

## Re-render

From `composition/`:

```bash
HYPERFRAMES_RENDER_DETACHED=1 npx hyperframes render \
  --output ../brag.mp4 \
  --low-memory-mode -w 1
```
