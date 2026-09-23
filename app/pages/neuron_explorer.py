"""Neuron Explorer page — search, inspect, and explore individual neurons."""

import streamlit as st
import pandas as pd
from app.services import get_store
from app.theme import inject_css, render_section_label, render_badge, render_monospace, render_disclaimer

def render(demo_id=None):
    inject_css()
    store = get_store()
    df = store.neuron_table

    st.markdown("# Neuron Explorer")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Search and inspect individual neurons in the FlyWire connectome.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Search section
    st.markdown(render_section_label("SEARCH"), unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        search_val = st.text_input(
            "Enter neuron root_id",
            value=str(demo_id) if demo_id else "",
            placeholder="e.g. 720575940602380768",
            label_visibility="collapsed",
        )
    with col2:
        search_type = st.radio("Search by", ["root_id", "name"], horizontal=True, label_visibility="collapsed")

    if not search_val:
        st.markdown("""
        <div style="text-align:center; padding:80px 0; color:#6B7180">
            <div style="font-size:48px; margin-bottom:16px">🔍</div>
            <div style="font-size:16px">Search a neuron ID to begin exploring</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Resolve neuron
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

    # Two-column layout
    left, right = st.columns([3, 2])

    with left:
        # Identity
        st.markdown(render_section_label("IDENTITY"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="flymind-card">
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px">
                {render_badge(result.super_class or 'Unknown', 'teal')}
                {render_badge(result.nt_type or 'Unknown NT', 'violet' if result.nt_type else 'muted')}
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#6B7180; margin-bottom:4px">
                ROOT ID
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:16px; color:#EDEFF4; font-weight:500; margin-bottom:12px">
                {rid}
            </div>
            <div style="font-size:14px; color:#A6ADBB; margin-bottom:4px">
                <strong>Name:</strong> {result.name if pd.notna(result.name) else '—'}
            </div>
            <div style="font-size:14px; color:#A6ADBB; margin-bottom:4px">
                <strong>Primary Type:</strong> {result.primary_type or '—'}
            </div>
            <div style="font-size:14px; color:#A6ADBB">
                <strong>Flow:</strong> {result.flow or '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Neurotransmitter Profile
        st.markdown(render_section_label("NEUROTRANSMITTER PROFILE"), unsafe_allow_html=True)
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

        st.markdown(render_disclaimer("14.1% of neurons lack neurotransmitter annotations. The nt_type class is predicted, not measured."), unsafe_allow_html=True)

        # Morphology
        st.markdown(render_section_label("MORPHOLOGY"), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        for i, (key, label) in enumerate([("length_nm", "Length"), ("area_nm", "Area"), ("size_nm", "Size")]):
            val = getattr(result, key, None)
            if pd.notna(val) and val > 0:
                [m1, m2, m3][i].metric(label, f"{val:,.0f} nm")
            else:
                [m1, m2, m3][i].metric(label, "—")

        # Location
        st.markdown(render_section_label("LOCATION"), unsafe_allow_html=True)
        x, y, z = st.columns(3)
        for i, (key, label) in enumerate([("coord_x", "X"), ("coord_y", "Y"), ("coord_z", "Z")]):
            val = getattr(result, key, None)
            if pd.notna(val):
                [x, y, z][i].metric(label, f"{val:,.1f}")
            else:
                [x, y, z][i].metric(label, "—")

    with right:
        # Connectivity Summary
        st.markdown(render_section_label("CONNECTIVITY SUMMARY"), unsafe_allow_html=True)
        summary = store.get_connectivity_summary(rid)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Out", summary.outgoing_count)
        c2.metric("In", summary.incoming_count)
        c3.metric("Syn Out", f"{summary.total_synapses_out:,}")
        c4.metric("Syn In", f"{summary.total_synapses_in:,}")

        # Top outgoing
        st.markdown(render_section_label("TOP OUTGOING CONNECTIONS"), unsafe_allow_html=True)
        if summary.top_outgoing:
            for conn in summary.top_outgoing[:5]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:#12161F; border:1px solid #232A38; border-radius:8px; margin-bottom:4px">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#A6ADBB">{conn.target_root_id}</span>
                    <span style="font-size:12px; color:#6B7180">w={conn.weight}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No outgoing connections.")

        # Top incoming
        st.markdown(render_section_label("TOP INCOMING CONNECTIONS"), unsafe_allow_html=True)
        if summary.top_incoming:
            for conn in summary.top_incoming[:5]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:#12161F; border:1px solid #232A38; border-radius:8px; margin-bottom:4px">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#A6ADBB">{conn.source_root_id}</span>
                    <span style="font-size:12px; color:#6B7180">w={conn.weight}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No incoming connections.")

        # Quick actions
        st.markdown(render_section_label("ACTIONS"), unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Rank candidates", use_container_width=True):
                st.session_state["candidate_source"] = str(rid)
                st.rerun()
        with a2:
            if st.button("Predict connection", use_container_width=True):
                st.session_state["predictor_source"] = str(rid)
                st.rerun()
