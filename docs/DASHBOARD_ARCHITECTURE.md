# FlyMind Dashboard Architecture

## Overview

The FlyMind Interactive Research Dashboard is a Streamlit application that provides an interactive interface to the FlyMind connectome research project. It consumes the Phase 2 inference layer and Phase 3 presentation data layer.

---

## 1. Architecture

```
Streamlit UI (app/)
      |
      +-- services.py (cached DataStore singleton)
      |
      v
Presentation Data Layer (src/link_prediction/presentation.py)
      |
      v
Inference Layer (src/link_prediction/inference.py)
      |
      v
Validated Model / Research Artifacts
```

The Streamlit UI does NOT directly implement feature engineering, model loading, candidate exclusion, negative sampling, or prediction logic.

---

## 2. Directory Structure

```
app/
    __init__.py
    streamlit_app.py          # Entry point, sidebar navigation
    services.py               # Cached DataStore singleton
    pages/
        __init__.py
        overview.py           # Project overview, key results
        neuron_explorer.py    # Inspect individual neurons
        connection_predictor.py  # Score directed pairs
        candidate_ranking.py  # Rank candidate targets
        research_results.py   # Experiment results, calibration
        about.py              # Limitations, terminology
```

---

## 3. Pages

### Overview
- Research question
- Dataset statistics (139K neurons, 3.73M edges)
- Architecture diagram
- Key result: RF cold-start ROC-AUC = 0.980
- Scientific disclaimer

### Neuron Explorer
- Search by root_id
- Identity, morphology, location, NT profile
- Connectivity summary (in/out degree, synapse counts)
- Top outgoing/incoming connections
- Local neighborhood visualization

### Connection Predictor
- Source/target pair input
- Connection status (observed/not observed)
- Model ranking score
- Source vs target comparison
- Global feature importance

### Candidate Ranking
- Source neuron input
- Configurable K (10/25/50/100)
- Ranked candidates table
- Score distribution chart
- Candidate detail view

### Research Results
- Experiment 2A: Random edge holdout
- Experiment 2B: Cold-start evaluation
- Experiment 3: Robustness, calibration, biological patterns
- Feature importance visualization

### About / Limitations
- Dataset limitations
- Model limitations
- Evaluation limitations
- Biological limitations
- Scientific terminology guide
- How to run locally

---

## 4. Data Flow

1. User navigates to a page via sidebar
2. Page calls `get_store()` to get cached DataStore
3. DataStore delegates to Inference Layer for model operations
4. DataStore loads edges/results lazily on first use
5. Page renders results with Streamlit components

---

## 5. Caching Strategy

- `@st.cache_resource` on `get_store()` ensures DataStore loads once per session
- DataStore lazy-loads edges and results on first use
- Inference layer lazy-loads RF model and features on first use
- No redundant data loading across page navigations

---

## 6. Memory Considerations

- DataStore + Inference: ~400 MB peak
- Streamlit overhead: ~100 MB
- Total: ~500 MB (well under 5.5 GB limit)
- No all-pairs enumeration
- Bounded neighborhood queries (max_nodes parameter)

---

## 7. Scientific Terminology

All pages use consistent terminology:
- "model-suggested candidate connection" (not "discovered connection")
- "ranking score" (not "biological probability")
- "observed in evaluated dataset" (not "confirmed connection")
- "hypotheses for further investigation" (not "new discoveries")

---

## 8. How to Run

```bash
cd flymind
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501
