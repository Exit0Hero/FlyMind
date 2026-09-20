"""FlyMind Dashboard — About / Limitations Page."""

import streamlit as st


def render():
    st.title("About / Limitations")
    st.markdown("Transparent description of the FlyMind project, its methods, and limitations.")

    # --- Dataset ---
    st.markdown("### Dataset Limitations")
    st.markdown("""
    - **FlyWire FAFB** connectome: 139,255 neurons, 3.73M directed edges
    - **Incomplete annotations**: ~19,000 neurons lack neurotransmitter labels
    - **Dataset-specific**: Results apply to the evaluated Princeton connectome source
    - **Structural only**: No behavioral or functional data included
    """)

    # --- Model ---
    st.markdown("### Model Limitations")
    st.markdown("""
    - **Random Forest** is the primary validated link-ranking model (100 estimators, 60 pair features)
    - **GNN experiments** were comparative and did not outperform RF on cold-start evaluation
    - **Ranking scores** represent relative likelihood, not biological probability
    - **Predictions do not establish biological truth** — they are hypotheses for further investigation
    - **Candidate generation** uses random sampling, not exhaustive evaluation of all pairs
    """)

    # --- Evaluation ---
    st.markdown("### Evaluation Limitations")
    st.markdown("""
    - **Random-edge evaluation** (Experiment 2A) can be optimistic because nodes remain visible through other edges
    - **Cold-start evaluation** (Experiment 2B) tests a stricter generalization scenario where test nodes have zero training connectivity
    - **Experiment 2C** (neural network ablation) was aborted due to memory constraints and must NOT be represented as a scientific result
    - **Candidate sampling** means not all possible pairs are evaluated
    """)

    # --- Biological ---
    st.markdown("### Biological Limitations")
    st.markdown("""
    - **Unknown neurotransmitter annotations** are common (~97.3% of top candidates)
    - **Model associations are not causal mechanisms** — the model learns statistical patterns, not biological rules
    - **Unobserved does not mean biologically absent** — the connectome dataset is incomplete
    - **No behavioral prediction** — the model does not predict fly behavior
    """)

    # --- Terminology ---
    st.markdown("### Scientific Terminology")
    st.markdown("""
    **Preferred terms:**
    - model-suggested candidate connection
    - candidate connection hypothesis
    - ranking score
    - observed in evaluated dataset
    - not observed in evaluated dataset
    - generalizes to previously unseen neurons

    **Avoided terms:**
    - discovered connection
    - confirmed new connection
    - guaranteed connection
    - biological probability
    - proves / understands the brain
    - predicts fly behavior
    """)

    # --- How to Run ---
    st.markdown("### How to Run Locally")
    st.code("""
# From the flymind repository root:
pip install -r requirements.txt
streamlit run app/streamlit_app.py
    """)

    # --- Credits ---
    st.markdown("### Credits")
    st.markdown("""
    - **Dataset**: FlyWire consortium (FAFB connectome)
    - **ML**: scikit-learn (Random Forest), PyTorch Geometric (GraphSAGE)
    - **Dashboard**: Streamlit
    - **Research**: FlyMind project
    """)
