"""FlyMind Dashboard — Candidate Ranking Page."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render():
    st.title("Candidate Ranking")
    st.markdown("Rank model-suggested candidate targets for a source neuron.")

    st.markdown("""
    <div class="disclaimer">
    <strong>Hypotheses, not discoveries.</strong> These are neuron pairs ranked
    highly by the validated model among the evaluated candidate set. They are
    hypotheses for further investigation, not confirmed biological discoveries.
    </div>
    """, unsafe_allow_html=True)

    store = get_store()

    # --- Input ---
    col1, col2 = st.columns([2, 1])
    with col1:
        source_input = st.text_input(
            "Source neuron (presynaptic)",
            value="720575940596125868",
        )
    with col2:
        k = st.selectbox("Top K candidates", [10, 25, 50, 100], index=0)

    try:
        source_id = int(source_input)
    except ValueError:
        st.error("Please enter a valid integer root_id.")
        return

    # Validate source exists
    try:
        source_neuron = store.get_neuron(source_id)
    except KeyError:
        st.error(f"Source neuron {source_id} not found.")
        return

    # --- Rank candidates ---
    if st.button("Rank Candidates", type="primary"):
        with st.spinner("Ranking candidates (sampling + RF inference)..."):
            candidates = store.rank_candidate_targets(source_id, k=k)

        if not candidates:
            st.warning("No candidates found.")
            return

        st.success(f"Ranked {len(candidates)} model-suggested candidate targets.")

        # --- Source info ---
        st.markdown(f"### Source: `{source_id}`")
        cols = st.columns(4)
        cols[0].metric("Super Class", source_neuron.super_class or "Unknown")
        cols[1].metric("NT Type", source_neuron.nt_type or "Unknown")
        cols[2].metric("Length", f"{source_neuron.length_nm:,.0f} nm" if source_neuron.length_nm else "N/A")
        cols[3].metric("Primary Type", source_neuron.primary_type or "Unknown")

        # --- Candidates table ---
        st.markdown("### Model-Suggested Candidate Connections")

        rows = []
        for c in candidates:
            rows.append({
                "Rank": c.rank,
                "Target Root ID": c.target_root_id,
                "Score": c.score,
                "Target NT Type": c.target_nt_type or "Unknown",
                "Target Super Class": c.target_super_class or "Unknown",
                "Target Primary Type": c.target_primary_type or "Unknown",
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({"Score": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

        # --- Score distribution chart ---
        st.markdown("### Score Distribution")
        st.bar_chart(df.set_index("Rank")["Score"])

        # --- Candidate detail view ---
        st.markdown("### Candidate Detail")
        st.markdown("Select a candidate to inspect source vs target properties.")

        selected_rank = st.selectbox(
            "Select candidate rank",
            options=[c.rank for c in candidates],
            format_func=lambda r: f"Rank {r}: {candidates[r-1].target_root_id} (score={candidates[r-1].score:.4f})",
        )

        selected = candidates[selected_rank - 1]
        target_neuron = store.get_neuron(selected.target_root_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source**")
            st.markdown(f"- Root ID: `{source_id}`")
            st.markdown(f"- Super Class: {source_neuron.super_class or 'Unknown'}")
            st.markdown(f"- NT Type: {source_neuron.nt_type or 'Unknown'}")
            st.markdown(f"- Primary Type: {source_neuron.primary_type or 'Unknown'}")
            st.markdown(f"- Length: {source_neuron.length_nm:,.0f} nm" if source_neuron.length_nm else "- Length: N/A")
            st.markdown(f"- Coords: ({source_neuron.coord_x:.0f}, {source_neuron.coord_y:.0f}, {source_neuron.coord_z:.0f})"
                       if source_neuron.coord_x else "- Coords: N/A")

        with col2:
            st.markdown("**Target**")
            st.markdown(f"- Root ID: `{selected.target_root_id}`")
            st.markdown(f"- Super Class: {target_neuron.super_class or 'Unknown'}")
            st.markdown(f"- NT Type: {target_neuron.nt_type or 'Unknown'}")
            st.markdown(f"- Primary Type: {target_neuron.primary_type or 'Unknown'}")
            st.markdown(f"- Length: {target_neuron.length_nm:,.0f} nm" if target_neuron.length_nm else "- Length: N/A")
            st.markdown(f"- Coords: ({target_neuron.coord_x:.0f}, {target_neuron.coord_y:.0f}, {target_neuron.coord_z:.0f})"
                       if target_neuron.coord_x else "- Coords: N/A")

        st.markdown(f"**Ranking Score:** `{selected.score:.4f}`")
