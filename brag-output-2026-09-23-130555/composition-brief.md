# Hyperframes Composition Brief: FlyMind

## Objective
Create a short launch-style brag video for FlyMind — a dark neuro-lab instrument reveal showing the predictor scoring a pair and the candidate table ranking targets.

## Output
- Composition directory: `/home/akkushon-kamen/flymind/brag-output-2026-09-23-130555/composition/`
- Rendered video: `/home/akkushon-kamen/flymind/brag-output-2026-09-23-130555/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 19 seconds

## Source Material
- Project root: `/home/akkushon-kamen/flymind`
- Primary files read: `README.md`, `frontend/app/page.tsx`, `frontend/app/predictor/page.tsx`, `frontend/app/candidates/page.tsx`, `frontend/app/globals.css`, `frontend/app/layout.tsx`
- Product name: FlyMind
- Tagline / strongest claim: Cold-start ROC-AUC **0.9800**; "Machine learning for exploring patterns in neuron connectivity."
- Key UI or visual moment to recreate: Connection Predictor result card (giant violet score **71**, ScoreBar, "Observed in dataset" badge) and Candidate Ranking table (3 rows, 96%/94%/90%)
- Copy that must appear verbatim:
  - FlyMind
  - 139,255 neurons. Which ones connect?
  - Run Prediction
  - Model-Suggested Connection Score
  - Observed in dataset
  - Candidate Ranking
  - ROC-AUC 0.9800
  - PR-AUC 0.9739
  - Model-suggested candidate connections
  - flymind-one.vercel.app

## Creative Direction
- Tone preset: polished
- Creative direction: dark neuro-lab instrument launch — quiet premium product film
- Interpretation: 4 scenes, longer holds, soft slides/crossfades; product UI is the hero; no parody energy; readable scientific confidence.
- Angle: Show the working app: pick neurons → Predict → score 71 lands → candidates rank → proof metrics → logo. Earnest research tool, premium lab aesthetic.
- Hook: Gradient **FlyMind** ignites over faint teal grid with "139,255 neurons. Which ones connect?"
- Outro / punchline: FlyMind. Model-suggested connection scores for the connectome. flymind-one.vercel.app
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals / waveforms
  - Unrelated visual redesign (must match site palette)

## Visual Identity
- Background: `#0B0E14` (base); surfaces `#12161F`, `#1A1F2B`
- Text: `#EDEFF4` primary; `#A6ADBB` secondary; `#6B7180` muted
- Accent: `#4FD1C5` teal; secondary `#8B7CF6` violet; success `#3FBE8C`; warning `#E3B341`
- Display font: Inter Tight (fallback system-ui)
- Body font: Inter (fallback system-ui)
- Mono: JetBrains Mono for root IDs / mono numbers
- Visual references from the project:
  - `text-gradient-accent` teal→violet wordmark
  - Predictor result card layout (badge RF · 71%, ScoreBar, InfoRows)
  - Candidate table with rank circles, NT badges, score bars
  - StatCard metric strip (Neurons / Directed Connections / ROC-AUC / PR-AUC)
  - Faint radial teal/violet glows on body background

## Storyboard
Use the storyboard in `/home/akkushon-kamen/flymind/brag-output-2026-09-23-130555/brag-plan.md` as the creative contract.

Scene summary:
1. Hook: FlyMind ignites — 4s — gradient wordmark + "139,255 neurons. Which ones connect?" + pipeline chips
2. Predictor UI: score lands — 6s — source/target, Run Prediction click, spinner, giant 71 + score bar + Observed badge
3. Candidate ranking table — 5s — 3 rows sequential 96/94/90%, hold full set
4. Proof + outro — 4s — ROC-AUC 0.9800 / PR-AUC 0.9739 / 8.3×, then logo + tagline + URL

## Audio
- Audio role: warm professional bed with restrained motion-matched accents
- Audio arc: bed establishes under hook → UI clicks score the flow → light card hits on candidate rows → soft logo payoff → fade
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (copy into `composition/assets/music/`)
- Music treatment: start 0, volume ~0.32, fade-in ~0.5s, swell into outro, fade-out last ~1.5s
- Music cue guidance: bundled preset at `/home/akkushon-kamen/.agents/skills/brag/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json` — strongCues: 8.74s (score reveal), 13.11s (candidate rows), 17.47s (logo). Beat grid ~0.55s apart. Treat as optional timing hints.
- Audio-reactive treatment: subtle; teal/violet hero glow and score-card presence breathe with RMS/bass. No waveform/equalizer visuals.
- Audio-coupled moments:
  - Scene 2 — Run Prediction simulated click + score settle (near strong cue 8.74s if readable)
  - Scene 3 — 3 candidate rows sequential (start near 13.11s; every-other-beat if text holds need floor)
  - Scene 4 — logo lands near 17.47s if readable; metrics hold ≥1.2s
- SFX selection guidance: sparse polished layer — ui mouseclick on button, soft drop/impactSoft on score land, card-place on each row, soft bell on outro; prefer low/medium HF-risk files
- SFX analysis guidance: `/home/akkushon-kamen/.agents/skills/brag/assets/sfx/sfx-analysis.md`
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: copy chosen music + selected SFX into `<output-dir>/composition/assets/`

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, `hyperframes-cli`. /brag is its own workflow: do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo / launch-video workflow. Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:
- Show at least one real UI, copy, or visual element from the source project (predictor card + candidates table required).
- Keep all text readable in the final render (short label ~0.8s settled; sentence ~0.3s/word).
- Keep the video within 15-25 seconds (target 19s).
- Include the planned music/SFX layer.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet. Choose SFX after the visual animation exists.
- Major reveals may move toward nearby strong cues within about 0.15s. Smaller entrances may align to nearby beat points within about 0.10s. Use only 1-3 strong cue locks.
- Sequential candidate row TEXT must hold full reading time — do not outrun the beat grid.
- When music is present, extract audio data and wire at least one visual element to subtle RMS/bass reactivity (glow/card presence).
- Use local assets for audio.
- Run `hyperframes check` before render — it is brag's single gate.
