"""FlyMind Dashboard — Connection Predictor Page."""

import streamlit as st
import pandas as pd
import numpy as np
from app.services import get_store
from src.link_prediction.presentation import FEATURE_GROUPS


def render():
    st.title("Connection Predictor")
    st.markdown("Score a directed neuron pair for connectivity likelihood.")

    store = get_store()

    # --- Input ---
    col1, col2 = st.columns(2)
    with col1:
        source_input = st.text_input(
            "Source neuron (presynaptic)",
            value="720575940596125868",
        )
    with col2:
        target_input = st.text_input(
            "Target neuron (postsynaptic)",
            value="720575940605825666",
        )

    try:
        source_id = int(source_input)
        target_id = int(target_input)
    except ValueError:
        st.error("Please enter valid integer root IDs.")
        return

    if st.button("Score Connection", type="primary"):
        # Validate neurons exist
        try:
            source_neuron = store.get_neuron(source_id)
        except KeyError:
            st.error(f"Source neuron {source_id} not found.")
            return
        try:
            target_neuron = store.get_neuron(target_id)
        except KeyError:
            st.error(f"Target neuron {target_id} not found.")
            return

        # Score the pair
        result = store._inference.score_connection(source_id, target_id)

        # --- Connection Status ---
        st.markdown("### Connection Status")
        if result.is_self_loop:
            st.warning("This is a self-loop (source == target).")
        elif result.is_known_edge:
            st.success("**Observed in evaluated dataset.**")
            st.markdown(
                "A connection between these neurons was observed in the "
                "Princeton connectome dataset."
            )
        else:
            st.info(
                "**Not observed in evaluated dataset.** No connection was "
                "observed for this pair in the evaluated dataset. This does "
                "not mean the neurons are definitely not connected biologically."
            )

        # --- Model Score ---
        st.markdown("### Model Ranking Score")
        st.metric("Score", f"{result.score:.4f}")

        st.markdown("""
        <div class="disclaimer">
        This score is intended for candidate ranking and should not be
        interpreted as a literal biological probability. It represents
        the model's assessment of relative connectivity likelihood based
        on neuron-level properties.
        </div>
        """, unsafe_allow_html=True)

        # --- Source Properties ---
        st.markdown("### Source vs Target Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source**")
            st.markdown(f"- Root ID: `{source_id}`")
            st.markdown(f"- Super Class: {source_neuron.super_class or 'Unknown'}")
            st.markdown(f"- Primary Type: {source_neuron.primary_type or 'Unknown'}")
            st.markdown(f"- NT Type: {source_neuron.nt_type or 'Unknown'}")
            st.markdown(f"- Length: {source_neuron.length_nm:,.0f} nm" if source_neuron.length_nm else "- Length: N/A")
            st.markdown(f"- Area: {source_neuron.area_nm:,.0f} nm²" if source_neuron.area_nm else "- Area: N/A")
            st.markdown(f"- Coords: ({source_neuron.coord_x:.0f}, {source_neuron.coord_y:.0f}, {source_neuron.coord_z:.0f})"
                       if source_neuron.coord_x else "- Coords: N/A")

        with col2:
            st.markdown("**Target**")
            st.markdown(f"- Root ID: `{target_id}`")
            st.markdown(f"- Super Class: {target_neuron.super_class or 'Unknown'}")
            st.markdown(f"- Primary Type: {target_neuron.primary_type or 'Unknown'}")
            st.markdown(f"- NT Type: {target_neuron.nt_type or 'Unknown'}")
            st.markdown(f"- Length: {target_neuron.length_nm:,.0f} nm" if target_neuron.length_nm else "- Length: N/A")
            st.markdown(f"- Area: {target_neuron.area_nm:,.0f} nm²" if target_neuron.area_nm else "- Area: N/A")
            st.markdown(f"- Coords: ({target_neuron.coord_x:.0f}, {target_neuron.coord_y:.0f}, {target_neuron.coord_z:.0f})"
                       if target_neuron.coord_x else "- Coords: N/A")

        # --- Feature Importance (global) ---
        st.markdown("### Global Feature Importance")
        st.markdown(
            "Global model feature importance is shown below; this describes "
            "overall model behavior and is not a local explanation for this pair."
        )

        fi = store.get_feature_importance()
        if fi:
            fi_df = pd.DataFrame([
                {"Feature": f.feature, "Importance": f.importance, "Group": f.group}
                for f in fi
            ])
            st.dataframe(fi_df.style.format({"Importance": "{:.4f}"}), hide_index=True)

            # Bar chart
            st.bar_chart(fi_df.set_index("Feature")["Importance"])
