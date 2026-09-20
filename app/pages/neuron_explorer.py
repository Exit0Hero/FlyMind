"""Neuron Explorer page — search, inspect, and explore individual neurons."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render(demo_id=None):
    store = get_store()
    df = store.neuron_table

    st.markdown("# Neuron Explorer")
    st.markdown("Search and inspect individual neurons in the FlyWire connectome.")

    # --- Search ---
    st.markdown("### Search Neuron")
    col1, col2 = st.columns([2, 1])
    with col1:
        search_val = st.text_input(
            "Enter neuron root_id",
            value=str(demo_id) if demo_id else "",
            placeholder="e.g. 720575940602380768",
        )
    with col2:
        search_type = st.radio("Search by", ["root_id", "name"], horizontal=True)

    if not search_val:
        st.info("Enter a neuron ID or name to begin exploring.")
        return

    if search_type == "root_id":
        try:
            rid = int(search_val)
        except ValueError:
            st.error("Invalid root_id. Must be an integer.")
            return
        result = store.get_neuron(rid)
        if result is None:
            st.error(f"Neuron {rid} not found.")
            return
    else:
        matches = df[df["name"].str.contains(search_val, case=False, na=False)]
        if matches.empty:
            st.error(f"No neuron name matches '{search_val}'.")
            return
        rid = int(matches.iloc[0]["root_id"])
        result = store.get_neuron(rid)
        if result is None:
            st.error(f"Neuron {rid} not found.")
            return

    # --- Neuron Identity ---
    st.markdown("---")
    st.markdown("## Neuron Identity")
    id_col1, id_col2, id_col3, id_col4 = st.columns(4)
    id_col1.metric("Root ID", str(rid))
    name_val = result.name if pd.notna(result.name) else "—"
    id_col2.metric("Name", name_val)
    id_col3.metric("Super-class", result.super_class or "—")
    id_col4.metric("Primary Type", result.primary_type or "—")

    # --- Neurotransmitter Profile ---
    st.markdown("## Neurotransmitter Profile")
    nt = result.nt_type
    if pd.notna(nt):
        st.markdown(f"**Class:** `{nt}`")
    else:
        st.markdown("**Class:** Unknown")

    # NT sub-scores
    nt_metrics = []
    for nt_key, label in [("ach_avg", "ACH"), ("gaba_avg", "GABA"),
                           ("glut_avg", "GLUT"), ("da_avg", "DA"),
                           ("ser_avg", "SER"), ("oct_avg", "OCT")]:
        val = getattr(result, nt_key, None)
        if pd.notna(val) and val > 0:
            nt_metrics.append((label, float(val)))
    if nt_metrics:
        nt_metrics.sort(key=lambda x: x[1], reverse=True)
        cols = st.columns(min(len(nt_metrics), 6))
        for i, (label, val) in enumerate(nt_metrics):
            cols[i].metric(label, f"{val:.4f}")
    else:
        st.caption("No sub-type scores available.")

    st.markdown("""
    <div class="disclaimer">
    <strong>Note:</strong> 97.3% of neurons have unknown neurotransmitter annotations.
    The "nt_type" class is predicted, not measured.
    </div>
    """, unsafe_allow_html=True)

    # --- Morphology ---
    st.markdown("## Morphology")
    morph_cols = st.columns(3)
    for i, (key, label) in enumerate([
        ("length_nm", "Length"),
        ("area_nm", "Area"),
        ("size_nm", "Size (soma)"),
    ]):
        val = getattr(result, key, None)
        if pd.notna(val) and val > 0:
            morph_cols[i].metric(label, f"{val:,.0f} nm")
        else:
            morph_cols[i].metric(label, "—")

    # --- Location ---
    st.markdown("## Location (Centroid Coordinates)")
    coord_cols = st.columns(3)
    for i, (key, label) in enumerate([
        ("coord_x", "X"),
        ("coord_y", "Y"),
        ("coord_z", "Z"),
    ]):
        val = getattr(result, key, None)
        if pd.notna(val):
            coord_cols[i].metric(label, f"{val:,.1f}")
        else:
            coord_cols[i].metric(label, "—")

    # --- Connectivity Summary ---
    st.markdown("## Connectivity Summary")
    summary = store.get_connectivity_summary(rid)
    conn_cols = st.columns(4)
    conn_cols[0].metric("Outgoing", summary.outgoing_count)
    conn_cols[1].metric("Incoming", summary.incoming_count)
    conn_cols[2].metric("Total Synapses Out", summary.total_synapses_out)
    conn_cols[3].metric("Total Synapses In", summary.total_synapses_in)

    # --- Local Neighborhood ---
    st.markdown("## Local Connectivity Neighborhood")
    st.markdown("Visual distinction: <span class='observed'>● Observed</span> | "
                "<span class='candidate'>● Candidate (from local neighborhood)</span>",
                unsafe_allow_html=True)

    neighborhood = store.get_neighborhood(rid, max_depth=1)
    if not neighborhood["nodes"]:
        st.info("No connectivity data available for this neuron.")
        return

    st.json({
        "nodes": len(neighborhood["nodes"]),
        "edges": len(neighborhood["edges"]),
        "depth": neighborhood["depth"],
    })

    # Show edges in a table
    if neighborhood["edges"]:
        edge_df = pd.DataFrame(neighborhood["edges"])
        edge_df["from"] = edge_df["from"].astype(str)
        edge_df["to"] = edge_df["to"].astype(str)
        st.dataframe(edge_df, use_container_width=True, height=300)

    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Rank candidates for this neuron", use_container_width=True):
            st.session_state["candidate_source"] = str(rid)
            st.switch_page("app/streamlit_app.py")
    with a2:
        if st.button("Check connection from this neuron", use_container_width=True):
            st.session_state["predictor_source"] = str(rid)
            st.switch_page("app/streamlit_app.py")
