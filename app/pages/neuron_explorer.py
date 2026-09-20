"""FlyMind Dashboard — Neuron Explorer Page."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render():
    st.title("Neuron Explorer")
    st.markdown("Inspect individual neurons and their connectivity patterns.")

    store = get_store()

    # --- Input ---
    root_id_input = st.text_input(
        "Enter neuron root_id",
        value="720575940596125868",
        help="FlyWire root ID (uint64)",
    )

    try:
        root_id = int(root_id_input)
    except ValueError:
        st.error("Please enter a valid integer root_id.")
        return

    try:
        neuron = store.get_neuron(root_id)
    except KeyError:
        st.error(f"Neuron {root_id} not found in dataset.")
        return

    # --- Identity ---
    st.markdown("### Identity")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Root ID", str(neuron.root_id))
    col2.metric("Super Class", neuron.super_class or "Unknown")
    col3.metric("Primary Type", neuron.primary_type or "Unknown")
    col4.metric("Name", neuron.name or "N/A")

    # --- Morphology ---
    st.markdown("### Morphology")
    col1, col2, col3 = st.columns(3)
    col1.metric("Length (nm)", f"{neuron.length_nm:,.0f}" if neuron.length_nm else "N/A")
    col2.metric("Area (nm²)", f"{neuron.area_nm:,.0f}" if neuron.area_nm else "N/A")
    col3.metric("Size (nm³)", f"{neuron.size_nm:,.0f}" if neuron.size_nm else "N/A")

    # --- Location ---
    st.markdown("### Location")
    col1, col2, col3 = st.columns(3)
    col1.metric("X", f"{neuron.coord_x:,.0f}" if neuron.coord_x else "N/A")
    col2.metric("Y", f"{neuron.coord_y:,.0f}" if neuron.coord_y else "N/A")
    col3.metric("Z", f"{neuron.coord_z:,.0f}" if neuron.coord_z else "N/A")

    # --- Neurotransmitter Profile ---
    st.markdown("### Neurotransmitter Profile")
    st.markdown(
        "Values indicate predicted neurotransmitter expression levels. "
        "Unknown annotations are preserved explicitly."
    )

    nt_data = {
        "Type": ["ACH", "GABA", "GLUT", "DA", "SER", "OCT", "Score"],
        "Value": [
            neuron.nt_type or "Unknown",
            "N/A (use GABA avg)",
            "N/A (use GLUT avg)",
            "N/A (use DA avg)",
            "N/A (use SER avg)",
            "N/A (use OCT avg)",
            f"{neuron.nt_type_score:.3f}" if neuron.nt_type_score else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(nt_data), hide_index=True, use_container_width=True)

    # --- Connectivity Summary ---
    st.markdown("### Connectivity Summary")
    summary = store.get_connectivity_summary(root_id, top_k=5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Outgoing Connections", summary.outgoing_count)
    col2.metric("Incoming Connections", summary.incoming_count)
    col3.metric("Total Synapses Out", f"{summary.total_synapses_out:,}")
    col4.metric("Total Synapses In", f"{summary.total_synapses_in:,}")

    # --- Top Outgoing ---
    if summary.top_outgoing:
        st.markdown("### Top Outgoing Connections")
        st.caption("Observed connections from this neuron, sorted by synapse count.")
        rows = []
        for c in summary.top_outgoing:
            rows.append({
                "Target": c.target_root_id,
                "Synapses": c.weight,
                "NT Type": c.target_nt_type or "Unknown",
                "Super Class": c.target_super_class or "Unknown",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- Top Incoming ---
    if summary.top_incoming:
        st.markdown("### Top Incoming Connections")
        st.caption("Observed connections to this neuron, sorted by synapse count.")
        rows = []
        for c in summary.top_incoming:
            rows.append({
                "Source": c.source_root_id,
                "Synapses": c.weight,
                "NT Type": c.source_nt_type or "Unknown",
                "Super Class": c.source_super_class or "Unknown",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- Local Neighborhood ---
    st.markdown("### Local Neighborhood")
    st.caption(
        "Limited local neighborhood visualization. "
        "Observed connections are <span class='observed'>green</span>. "
        "Model-suggested candidates are <span class='candidate'>orange</span>.",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        direction = st.selectbox("Direction", ["outgoing", "incoming", "both"], index=0)
    with col2:
        max_nodes = st.selectbox("Max nodes", [10, 25, 50], index=1)
    with col3:
        include_candidates = st.checkbox("Include model-suggested candidates", value=False)

    hood = store.get_neighborhood(
        root_id,
        direction=direction,
        max_nodes=max_nodes,
        include_candidates=include_candidates,
        candidate_k=10,
    )

    if hood.edges:
        # Show nodes table
        nodes_df = pd.DataFrame(hood.nodes)
        edges_df = pd.DataFrame(hood.edges)

        st.markdown(f"**{len(hood.nodes)} nodes, {len(hood.edges)} edges**")

        tab1, tab2 = st.tabs(["Nodes", "Edges"])
        with tab1:
            st.dataframe(nodes_df, hide_index=True, use_container_width=True)
        with tab2:
            st.dataframe(edges_df, hide_index=True, use_container_width=True)
    else:
        st.info("No connections found for this neuron.")
