"""Overview page — project summary and key metrics."""

import streamlit as st


def render():
    st.markdown("""
    # FlyMind

    **Learning neuron connectivity from biological and morphological properties**

    *Fruit-fly (Drosophila melanogaster) connectome research*
    """)

    # Research question
    st.markdown("""
    <div class="disclaimer-info">
    <strong>Research Question:</strong> Can neuron-level biological and morphological properties
    predict directed connectivity in the fruit-fly connectome, and do those relationships
    generalize to previously unseen neurons?
    </div>
    """, unsafe_allow_html=True)

    # Quick action buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("app/streamlit_app.py", label="Explore a Neuron", icon="🔍")
    with c2:
        st.page_link("app/streamlit_app.py", label="Rank Candidate Connections", icon="📊")
    with c3:
        st.page_link("app/streamlit_app.py", label="View Research Results", icon="📋")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Key metrics
    st.markdown("## Key Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Neurons", "139,255")
    m2.metric("Directed Edges", "3.73M")
    m3.metric("Cold-start ROC-AUC", "~0.98")
    m4.metric("Cold-start PR-AUC", "0.974")
    m5.metric("Recall@10", "0.814")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Architecture
    st.markdown("## Research Architecture")
    st.markdown("""
    ```
    FlyWire Connectome (FAFB)
                ↓
    Neuron Biological Features    +    Morphological Features
                ↓
         Feature Engineering
      (source, target, pair features)
                ↓
          Random Forest
        (validated primary model)
                ↓
        Connection Scoring
                ↓
       Candidate Ranking
                ↓
    Model-suggested candidate connections
    ```
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # What can I interact with
    st.markdown("## What Can I Interact With?")
    st.markdown("""
    | Page | What it does |
    |------|-------------|
    | **Neuron Explorer** | Search any neuron by ID; view morphology, location, neurotransmitter profile, connectivity summary, and local neighborhood |
    | **Connection Predictor** | Evaluate a specific source–target pair; see model ranking score and feature comparison |
    | **Candidate Ranking** | Rank all candidate target neurons for a source; inspect top candidates with side-by-side comparison |
    | **Research Results** | View validated experiment results: link prediction, cold-start generalization, calibration, feature importance |
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown("## How It Works")
    st.markdown("""
    1. Represent each neuron using biological and morphological features (15 base features)
    2. Construct source-target feature pairs (pair, difference, interaction — 60 dimensions)
    3. Train a validated Random Forest on observed connectivity
    4. Evaluate on held-out neurons (cold-start evaluation)
    5. Rank candidate target neurons for any source neuron
    6. Present high-ranking pairs as hypotheses for further investigation
    """)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>Important:</strong> Candidate rankings represent model-suggested connection hypotheses,
    not confirmed biological discoveries. A high ranking score indicates statistical association
    in the evaluated model, not proof that a biological connection exists.
    </div>
    """, unsafe_allow_html=True)
