"""FlyMind Dashboard — Overview Page."""

import streamlit as st
from app.services import get_store


def render():
    st.title("FlyMind")
    st.subheader("Learning neuron connectivity from biological and morphological properties")

    st.markdown("""
    ### Research Question
    > Can neuron-level biological and morphological properties predict
    > directed connectivity in the fruit-fly connectome, and do those
    > relationships generalize to previously unseen neurons?
    """)

    store = get_store()

    # --- Key dataset statistics ---
    st.markdown("### Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Neurons", f"{store.n_neurons:,}")
    col2.metric("Directed Edges", f"{store.n_edges:,}")
    col3.metric("Node Features", "15")
    col4.metric("Pair Features", "60")

    # --- Architecture ---
    st.markdown("""
    ### Architecture
    ```
    Neuron properties (morphology, spatial, neurotransmitter)
              |
    Feature engineering (pair features: [x_A, x_B, |x_A-x_B|, x_A*x_B])
              |
    Validated Random Forest model (100 estimators)
              |
    Connection scoring (ranking score 0-1)
              |
    Candidate ranking (model-suggested hypotheses)
    ```
    """)

    # --- Key Results ---
    st.markdown("### Key Research Results")

    st.markdown("#### Cold-Start Evaluation (strongest validation)")
    st.markdown(
        "Test neurons have **zero training connectivity information**. "
        "This tests whether node-level features generalize to unseen neurons."
    )

    comparison = store.get_model_comparison()
    if comparison:
        import pandas as pd
        df = pd.DataFrame(comparison)
        df = df[["model", "roc_auc", "pr_auc"]].rename(columns={
            "model": "Model",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
        })
        st.dataframe(df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}), hide_index=True)

    # --- Primary result highlight ---
    st.markdown("""
    ### Primary Result
    > **RF Node Features (cold-start): ROC-AUC = 0.980**
    >
    > Neuron-level features (morphology, spatial coordinates, neurotransmitter expression)
    > are sufficient to predict directed connections with high accuracy, and this
    > generalizes to previously unseen neurons.
    """)

    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer">
    <strong>Scientific Disclaimer:</strong> Candidate rankings represent model-suggested
    connection hypotheses. They do not establish that an unobserved biological connection
    exists. The model is a computational ranking system, not a biological probability estimator.
    </div>
    """, unsafe_allow_html=True)
