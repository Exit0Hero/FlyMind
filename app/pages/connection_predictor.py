"""Connection Predictor page — evaluate a specific source-target pair."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render():
    store = get_store()
    df = store.neuron_table

    st.markdown("# Connection Predictor")
    st.markdown("Evaluate a specific source-target neuron pair and see the model ranking score.")

    # --- Input ---
    st.markdown("### Select Neuron Pair")
    col1, col2 = st.columns(2)
    with col1:
        source_id = st.text_input("Source neuron root_id", placeholder="e.g. 720575940597856265")
    with col2:
        target_id = st.text_input("Target neuron root_id", placeholder="e.g. 720575940602380768")

    if not source_id or not target_id:
        st.info("Enter both source and target neuron IDs to evaluate.")
        return

    try:
        src = int(source_id)
        tgt = int(target_id)
    except ValueError:
        st.error("Both IDs must be integers.")
        return

    src_data = store.get_neuron(src)
    tgt_data = store.get_neuron(tgt)
    if src_data is None:
        st.error(f"Source neuron {src} not found.")
        return
    if tgt_data is None:
        st.error(f"Target neuron {tgt} not found.")
        return

    # --- Source vs Target comparison ---
    st.markdown("---")
    st.markdown("## Source vs Target Comparison")

    scol, tcol = st.columns(2)

    def _neuron_card(col, data, label):
        with col:
            st.markdown(f"**{label}**")
            name_val = data.name if pd.notna(data.name) else "—"
            st.markdown(f"**Root ID:** `{data.root_id}`")
            st.markdown(f"**Name:** {name_val}")
            st.markdown(f"**Super-class:** {data.super_class or '—'}")
            st.markdown(f"**Primary Type:** {data.primary_type or '—'}")

            nt = data.nt_type
            st.markdown(f"**Neurotransmitter:** `{nt if pd.notna(nt) else 'Unknown'}`")

            for key, label_m in [("length_nm", "Length"), ("area_nm", "Area"), ("size_nm", "Size")]:
                val = getattr(data, key, None)
                if pd.notna(val) and val > 0:
                    st.markdown(f"**{label_m}:** {val:,.0f} nm")
                else:
                    st.markdown(f"**{label_m}:** —")

            for key, label_c in [("coord_x", "X"), ("coord_y", "Y"), ("coord_z", "Z")]:
                val = getattr(data, key, None)
                if pd.notna(val):
                    st.markdown(f"**{label_c}:** {val:,.1f}")

    _neuron_card(scol, src_data, "SOURCE NEURON")
    _neuron_card(tcol, tgt_data, "TARGET NEURON")

    # --- Connection Status ---
    st.markdown("---")
    st.markdown("## Connection Status")
    is_observed = store.check_connection(src, tgt)
    if is_observed:
        st.success("**Observed in evaluated dataset** — This directed connection is present in the FlyWire connectivity graph used for evaluation.")
    else:
        st.info("**Not observed in evaluated dataset** — This directed connection is not present in the evaluated graph.")

    # --- Ranking Score ---
    st.markdown("## Model Ranking Score")
    score = store.score_connection(src, tgt)
    st.metric("Ranking Score", f"{score:.4f}")

    st.markdown("""
    <div class="disclaimer">
    <strong>Interpretation:</strong> This score is used for candidate ranking and should not be
    interpreted as a literal biological probability. A higher score indicates the model ranks
    this pair as more likely to have a directed connection based on learned statistical patterns.
    </div>
    """, unsafe_allow_html=True)

    # --- Pair Features ---
    st.markdown("## Pair-Level Model Features")
    src_arr = store.get_feature_vector(src)
    tgt_arr = store.get_feature_vector(tgt)
    if src_arr is not None and tgt_arr is not None:
        pair_data = {}
        feature_names = ["length_nm", "area_nm", "size_nm", "coord_x", "coord_y", "coord_z",
                         "nt_type_score", "ach_avg", "gaba_avg", "glut_avg", "da_avg", "ser_avg", "oct_avg",
                         "synapse_count", "total_nt_score"]
        for i, fname in enumerate(feature_names):
            pair_data[fname] = {
                "Source": f"{src_arr[i]:.4f}" if i < len(src_arr) else "—",
                "Target": f"{tgt_arr[i]:.4f}" if i < len(tgt_arr) else "—",
                "|Diff|": f"{abs(src_arr[i] - tgt_arr[i]):.4f}" if i < len(src_arr) and i < len(tgt_arr) else "—",
                "Product": f"{src_arr[i] * tgt_arr[i]:.4f}" if i < len(src_arr) and i < len(tgt_arr) else "—",
            }
        pair_df = pd.DataFrame(pair_data).T
        st.dataframe(pair_df, use_container_width=True)
    else:
        st.warning("Feature vectors not available.")

    # --- Quick actions ---
    st.markdown("---")
    st.markdown("### Quick Actions")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Rank candidates for this source", use_container_width=True):
            st.session_state["candidate_source"] = source_id
            st.switch_page("app/streamlit_app.py")
    with a2:
        if st.button("Explore source neuron in detail", use_container_width=True):
            st.session_state["explorer_id"] = source_id
            st.switch_page("app/streamlit_app.py")
