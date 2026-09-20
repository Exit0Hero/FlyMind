# FlyMind

**Learning neuron connectivity from biological and morphological properties**

FlyMind explores whether neuron-level biological and morphological properties can predict
directed connectivity in the fruit-fly (*Drosophila melanogaster*) connectome, and whether
those relationships generalize to previously unseen neurons.

---

## Research Question

Can neuron-level biological and morphological properties predict directed connectivity
in the fruit-fly connectome, and do those relationships generalize to previously unseen neurons?

## Key Findings

| Metric | Value |
|--------|-------|
| Neurons | 139,255 |
| Directed edges | 3,732,460 |
| Cold-start ROC-AUC | ~0.98 |
| Cold-start PR-AUC | 0.974 |
| Recall@10 | 0.814 |

**Cold-start generalization:** Training edges touching held-out neurons are excluded.
The model is evaluated entirely on previously unseen neurons.

## Architecture

```
FlyWire Connectome (FAFB)
            ↓
Neuron Biological Features  +  Morphological Features
            ↓
     Feature Engineering (60 dimensions)
            ↓
      Random Forest (validated primary model)
            ↓
    Connection Scoring → Candidate Ranking
            ↓
Model-suggested candidate connections
```

## Dashboard

Interactive research dashboard for exploring neuron connectivity:

```bash
streamlit run app/streamlit_app.py
```

**Pages:**
- **Overview** — Research question, key metrics, architecture
- **Neuron Explorer** — Search and inspect individual neurons
- **Connection Predictor** — Evaluate specific source-target pairs
- **Candidate Ranking** — Rank candidate target connections
- **Research Results** — Validated experiment results
- **About / Limitations** — Scientific context and known limitations

## Dataset

FlyWire FAFB (Full Adult Fly Brain) connectome:
- 139,255 neurons
- 3,732,460 unique directed edges
- 15 safe features (13 numeric + 2 categorical)
- 76 brain regions (neuropils)

**Citation:** If you use this data, please cite the FlyWire consortium and the original
FAFB dataset as described at [https://flywire.ai/](https://flywire.ai/).

## Experiments

| Experiment | Description | Key Result |
|------------|-------------|------------|
| **2A** | Random edge link prediction | ROC-AUC ~0.98 |
| **2B** | Candidate ranking | Recall@10 0.814 |
| **2C** | (Aborted — memory constraints) | N/A |
| **3** | Cold-start generalization | ROC-AUC ~0.98 on held-out neurons |

## Results

- **Random Forest** is the validated primary model
- **GraphSAGE** was evaluated comparatively and performed worse in these experiments
- **Cold-start evaluation** demonstrates generalization to previously unseen neurons
- **Feature importance:** morphology (area, length, size) and spatial features dominate

## Limitations

- **Dataset-specific:** Results apply to this specific FlyWire dataset
- **Not causal:** Model learns statistical associations, not causal mechanisms
- **Unknown annotations:** 97.3% of neurons have unknown neurotransmitter annotations
- **Requires validation:** Candidate connections require biological confirmation
- **Not behavior:** Does not predict neural circuits or fly behavior

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/streamlit_app.py

# Run tests
python -m pytest tests/ -v --ignore=tests/test_link_prediction.py --ignore=tests/test_cold_start.py
```

## Project Structure

```
flymind/
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py    # Entry point
│   ├── services.py         # Cached data services
│   └── pages/              # Dashboard pages
├── src/
│   └── link_prediction/    # Core ML pipeline
│       ├── config.py       # Paths, constants
│       ├── models.py       # RF training, GraphSAGE
│       ├── features.py     # Feature engineering
│       ├── inference.py    # Inference layer
│       └── presentation.py # Presentation data layer
├── models/                 # Trained models
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Processed data
├── results/
│   ├── figures/            # Generated plots
│   └── reports/            # JSON metrics
├── docs/                   # Documentation
│   ├── FINAL_RESEARCH_AUDIT.md
│   ├── FINAL_RESULTS.md
│   ├── INFERENCE_ARCHITECTURE.md
│   ├── PRESENTATION_DATA_LAYER.md
│   ├── DASHBOARD_ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── JUDGE_QA.md
└── tests/                  # Test suite
```

## Scientific Disclaimer

Candidate rankings represent **model-suggested connection hypotheses**, not confirmed
biological discoveries. A high ranking score indicates statistical association in the
evaluated model, not proof that a biological connection exists. Biological validation
requires wet-lab experiments.
