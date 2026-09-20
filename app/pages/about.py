"""About / Limitations page — scientific context and known limitations."""

import streamlit as st


def render():
    st.markdown("# About / Limitations")

    st.markdown("""
    FlyMind is a research tool for exploring neuron connectivity relationships in the
    Drosophila melanogaster connectome. This page describes what the system does and does not claim.
    """)

    st.markdown("---")

    # What FlyMind does NOT claim
    st.markdown("## What FlyMind Does NOT Claim")
    st.markdown("""
    <div class="disclaimer">
    <ul>
    <li>Prove an unobserved connection exists</li>
    <li>Discover biological truth automatically</li>
    <li>Predict fly behavior</li>
    <li>Establish neurotransmitter causality</li>
    <li>Reconstruct the entire connectome</li>
    <li>Provide literal biological probabilities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # What FlyMind DOES
    st.markdown("## What FlyMind DOES")
    st.markdown("""
    <div class="disclaimer-info">
    <ul>
    <li>Learn statistical relationships from the evaluated connectome</li>
    <li>Test generalization to held-out neurons</li>
    <li>Rank candidate connection hypotheses</li>
    <li>Provide an interactive research interface</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Limitations
    st.markdown("## Known Limitations")

    with st.expander("Dataset", expanded=True):
        st.markdown("""
        - 139,255 neurons from FlyWire FAFB connectome
        - 3,732,460 unique directed edges
        - Dataset-specific coverage and filtering
        - **14.1% of neurons (19,658) lack neurotransmitter annotations**
        - Annotation quality varies across brain regions
        """)

    with st.expander("Model", expanded=True):
        st.markdown("""
        - Random Forest is the validated primary model (100 estimators, 60 features)
        - GraphSAGE was evaluated comparatively and performed worse in these experiments
        - Model captures statistical correlations, not causal mechanisms
        - 15 base features + pair engineering (source, target, difference, product) = 60 dimensions
        """)

    with st.expander("Generalization", expanded=True):
        st.markdown("""
        - Cold-start evaluation provides evidence for generalization within the evaluated setup
        - Training edges touching held-out neurons are excluded
        - **Does not imply universal biological generalization**
        - Generalization is specific to this dataset, feature set, and evaluation protocol
        """)

    with st.expander("Biology", expanded=True):
        st.markdown("""
        - Model associations are not causal biological mechanisms
        - Neurotransmitter annotations are predicted, not measured for most neurons
        - Spatial proximity may confound connectivity predictions
        - Biological validation requires wet-lab experiments
        """)

    with st.expander("Candidate Interpretation", expanded=True):
        st.markdown("""
        - Candidate rankings are hypotheses for investigation
        - A high ranking score does not prove a biological connection exists
        - Ranking scores should not be interpreted as literal biological probabilities
        - Biological validation requires experimental confirmation
        """)

    with st.expander("Experiment 2C", expanded=True):
        st.markdown("""
        - Experiment 2C was aborted due to memory constraints (RAM exhaustion)
        - Not treated as a scientific result
        - No partial or incomplete result from Experiment 2C is presented
        """)

    st.markdown("---")

    # How it works (non-ML explanation)
    st.markdown("## How It Works (Non-Technical)")
    st.markdown("""
    1. **Represent** each neuron using biological and morphological features
    2. **Construct** source-target feature pairs
    3. **Train** a validated Random Forest on observed connectivity
    4. **Evaluate** on held-out neurons (cold-start)
    5. **Rank** candidate target neurons
    6. **Present** high-ranking pairs as hypotheses for further investigation
    """)

    # Terminology guide
    st.markdown("## Terminology Guide")
    st.markdown("""
    | Term | Meaning |
    |------|---------|
    | **Root ID** | Unique FlyWire neuron identifier |
    | **Super-class** | High-level biological classification (e.g., motor, sensory, central) |
    | **Primary Type** | Specific neuron type label |
    | **Ranking Score** | Model-derived score for candidate ordering |
    | **Cold-start** | Evaluation on previously unseen neurons |
    | **Observed connection** | Directed edge present in the evaluated graph |
    | **Candidate connection** | Model-suggested pair not present in the evaluated graph |
    | **Pair features** | Features constructed from source + target neuron properties |
    | **ECE** | Expected Calibration Error — measures probability calibration |
    | **Recall@K** | Fraction of true connections in top-K ranked candidates |
    """)

    st.markdown("---")

    # Dataset citation
    st.markdown("## Dataset Citation")
    st.markdown("""
    This project uses connectome data from [FlyWire](https://flywire.ai/), a complete wiring
    diagram of the adult fruit-fly brain (Drosophila melanogaster), built from the FAFB
    (Full Adult Fly Brain) electron microscopy volume.

    **Citation:** If you use this data, please cite the FlyWire consortium and the original
    FAFB dataset as described at [https://flywire.ai/](https://flywire.ai/).
    """)
