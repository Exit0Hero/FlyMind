# Brag Plan: FlyMind

## What is this app?
FlyMind predicts directed neuron connectivity in the fruit-fly connectome from 15 biological/morphological features — scoring any neuron pair and ranking candidate targets for researchers, with cold-start ROC-AUC 0.9800.

## The angle
A dark neuro-lab instrument reveal: show the actual product doing its thing — pick two neurons, hit Predict, watch a model-suggested score land — then rank a full candidate table in one motion. Earnest science tool, quietly premium, not a joke product.

## Hook (first 2-3 seconds)
On deep space-black, the gradient wordmark **FlyMind** ignites over a faint neural grid, with the line: "139,255 neurons. Which ones connect?"

## Key moments (the middle)
- Connection Predictor UI: source/target selected, **Run Prediction** click → giant violet **71 / 100** springs in with score bar + "Observed in dataset" badge
- Candidate Ranking table: 3 rows slam in one-by-one — rank chips, type, NT badge, score bars filling to **96% / 94% / 90%**
- Live metrics strip: **ROC-AUC 0.9800**, **PR-AUC 0.9739**, **8.3× better than random**

## Outro / punchline
"FlyMind." / "Model-suggested connection scores for the connectome." / flymind-one.vercel.app — teal→violet glow hold.

## User flow worth showing
1. Entry — Predictor page, source + target neuron selects filled
2. Key action — click **Run Prediction**, spinner beat
3. Result — score card lands (71, known edge), then candidates table ranks top targets

## Tone
- Preset: polished
- Creative direction: dark neuro-lab instrument launch — quiet premium product film
- Interpretation: fewer scenes, longer holds, soft slides/crossfades; confidence through restraint; no chaos, no parody winks; product UI carries the energy.

## Format: landscape — 1920x1080
## Duration: 19s

## Visual identity (from the project)
- Background: `#0B0E14` (base), surfaces `#12161F` / `#1A1F2B`
- Accent: `#4FD1C5` (teal), secondary `#8B7CF6` (violet), success `#3FBE8C`
- Text: `#EDEFF4` primary, `#A6ADBB` secondary, `#6B7180` muted
- Display font: Inter Tight
- Body font: Inter
- Mono: JetBrains Mono (root IDs, scores)
- Strongest visual element: teal→violet gradient wordmark + giant violet predictor score + dark card UI with neon accents

## Share copy (draft)
FlyMind scores any pair of neurons in the fruit-fly connectome — cold-start ROC-AUC 0.98, candidates ranked 8.3× better than random. Live at flymind-one.vercel.app

## Audio direction
- Role: warm professional bed with restrained motion-matched accents
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean — polished fit), ~110 BPM
- Music treatment: start 0, volume ~0.32, gentle fade-in over first 0.5s, hold under UI, slight swell into outro logo, fade-out last 1.5s
- Music cue guidance: preset at `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`; strongCues targets — **8.74s** (predictor score reveal), **13.11s** (candidate rows sequence start), **17.47s** (outro logo); beat-grid window ~13.1–14.2s for sequential candidate rows (snap every other beat if text holds need floor)
- Audio-reactive treatment: subtle; teal/violet hero glow and score-card presence breathe with RMS/bass — no waveform visuals
- SFX posture: sparse (3-5): mouse click on Run Prediction, soft drop/card for score landing, card-place for each candidate row, soft bell on outro
- Audio-coupled moments: Run click, score count-up landing, 3 candidate rows sequential, final logo
- Restraint rule: never louder than the UI clarity; no SFX over readable text reveals

## Storyboard

### Scene 1 — Hook: FlyMind ignites — 4s
Deep `#0B0E14` with faint teal grid/glow. Gradient **FlyMind** (Inter Tight, large) scales softly in. Subline holds: "139,255 neurons. Which ones connect?" Pipeline chips (FlyWire → Features → Random Forest → Ranking) fade in under, small.
Sequential/interaction: none required; optional chip fade stagger.
Audio intent: quiet confidence, bed establishes.
Audio-coupled idea: none mandatory; subtle glow breathes with music.
Music: vol-12 bed from 0.
Transition mood: soft crossfade → Scene 2

### Scene 2 — Predictor UI: score lands — 6s
Recreate Connection Predictor card (dark surface, violet accents): Source Neuron + Target Neuron fields show names/IDs; cursor moves to **Run Prediction**; spinner beat ~0.6s; result card springs in with giant violet **71**, "out of 100", score bar fills, badge **RF · 71%**, row "Observed in dataset".
Sequential/interaction: yes — fields present, button click simulated, then score number + badge + bar arrive as a set.
Audio intent: one crisp click on button; soft landing accent when 71 settles.
Audio-coupled idea: simulated click + score settle on strong cue ~8.74s if readable.
Music: bed continues; hit strong cue near score reveal.
Transition mood: clean slide/crossfade → Scene 3

### Scene 3 — Candidate ranking table — 5s
Recreate Candidate Ranking table header + 3 rows arriving one by one: rank chips 1/2/3, target types (LT42, LLPC3, …), NT badges, score bars filling to 96%, 94%, 90%. Footnote strip: "model-suggested candidate connections".
Sequential/interaction: yes — 3 rows one-by-one with card sounds; hold full set at end for readability.
Audio intent: light card/drop per row, restrained.
Audio-coupled idea: beat-grid rows near 13.11s strong cue; every-other-beat if needed for text hold.
Music: bed; lock row sequence start near 13.11s if it doesn't hurt reading.
Transition mood: soft crossfade → Scene 4

### Scene 4 — Proof + outro — 4s
Three metric tiles hold clean: **ROC-AUC 0.9800**, **PR-AUC 0.9739**, **8.3× vs random**. Then crossfade to centered **FlyMind** wordmark + tagline "Model-suggested connection scores for the connectome." + `flymind-one.vercel.app`. Teal→violet glow hold to end.
Sequential/interaction: metrics may arrive as a quick set (settled hold ≥1.2s); logo last.
Audio intent: one soft bell/announcement on logo; music fades under.
Audio-coupled idea: logo lands near strong cue ~17.47s if readable.
Music: swell then fade-out final ~1.5s.
Transition mood: end hold

**Music mood for this video:** steady, clean, professional (vol-12)
**Audio summary:** quiet bed establishes under the hook, restrained UI accents score the product flow, soft logo payoff, clean fade.
