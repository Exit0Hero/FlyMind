"""Connection Predictor page — evaluate a specific source-target pair."""

import streamlit as st
import pandas as pd
from app.services import get_store
from app.theme import inject_css, render_section_label, render_badge, render_score_bar, render_disclaimer, render_monospace

def render():
    inject_css()
    store = get_store()

    st.markdown("# Connection Predictor")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Evaluate a specific source-target neuron pair and see the model ranking score.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Input section
    st.markdown(render_section_label("SELECT NEURON PAIR"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        source_id = st.text_input("Source neuron root_id", placeholder="e.g. 720575940597856265", label_visibility="collapsed")
    with col2:
        target_id = st.text_input("Target neuron root_id", placeholder="e.g. 720575940602380768", label_visibility="collapsed")

    if not source_id or not target_id:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#6B7180">
            <div style="font-size:48px; margin-bottom:16px">🎯</div>
            <div style="font-size:16px">Enter both source and target neuron IDs to evaluate</div>
        </div>
        """, unsafe_allow_html=True)
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

    # Run prediction
    with st.spinner("Running prediction..."):
        score = store.score_connection(src, tgt)

    # Source → Target visual
    st.markdown("---")
    st.markdown(render_section_label("PREDICTION RESULT"), unsafe_allow_html=True)

    # Neuron cards with score between
    scol1, score_col, scol2 = st.columns([2, 1, 2])

    with scol1:
        st.markdown(f"""
        <div class="flymind-card">
            <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180; margin-bottom:8px">
                SOURCE NEURON
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#4FD1C5; margin-bottom:4px">
                {src}
            </div>
            <div style="font-size:14px; color:#EDEFF4; margin-bottom:4px">
                {src_data.primary_type or '—'}
            </div>
            <div style="font-size:12px; color:#6B7180">
                {src_data.super_class or '—'}
            </div>
            <div style="margin-top:8px">
                {render_badge(src_data.nt_type or 'Unknown', 'teal' if src_data.nt_type else 'muted')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with score_col:
        st.markdown(f"""
        <div style="text-align:center; padding:20px 0">
            <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180; margin-bottom:8px">
                MODEL SCORE
            </div>
            <div style="font-family:'Inter Tight',sans-serif; font-weight:700; font-size:48px; letter-spacing:-0.02em; color:#8B7CF6">
                {score.score:.4f}
            </div>
            <div style="margin-top:12px">
                {render_score_bar(score.score, 'predicted')}
            </div>
            <div style="margin-top:8px">
                {render_badge('PREDICTED', 'violet')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with scol2:
        st.markdown(f"""
        <div class="flymind-card">
            <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180; margin-bottom:8px">
                TARGET NEURON
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#8B7CF6; margin-bottom:4px">
                {tgt}
            </div>
            <div style="font-size:14px; color:#EDEFF4; margin-bottom:4px">
                {tgt_data.primary_type or '—'}
            </div>
            <div style="font-size:12px; color:#6B7180">
                {tgt_data.super_class or '—'}
            </div>
            <div style="margin-top:8px">
                {render_badge(tgt_data.nt_type or 'Unknown', 'teal' if tgt_data.nt_type else 'muted')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Connection status
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    if score.is_known_edge:
        st.markdown(f"""
        <div class="flymind-card" style="border-color:rgba(63,190,140,0.3)">
            <div style="display:flex; align-items:center; gap:8px">
                {render_badge('OBSERVED', 'success')}
                <span style="font-size:14px; color:#A6ADBB">
                    This directed connection is present in the FlyWire connectivity graph.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="flymind-card" style="border-color:rgba(139,124,246,0.3)">
            <div style="display:flex; align-items:center; gap:8px">
                {render_badge('NOT OBSERVED', 'violet')}
                <span style="font-size:14px; color:#A6ADBB">
                    This directed connection is not present in the evaluated graph.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(render_disclaimer(), unsafe_allow_html=True)

    # Feature comparison
    st.markdown("---")
    st.markdown(render_section_label("FEATURE COMPARISON"), unsafe_allow_html=True)
    src_arr = store.get_feature_vector(src)
    tgt_arr = store.get_feature_vector(tgt)
    if src_arr is not None and tgt_arr is not None:
        feature_names = ["nt_type_score", "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg",
                         "length_nm", "area_nm", "size_nm", "coord_x", "coord_y", "coord_z", "flow_enc", "side_x_enc"]
        pair_data = {}
        for i, fname in enumerate(feature_names):
            if i < len(src_arr) and i < len(tgt_arr):
                pair_data[fname] = {
                    "Source": f"{src_arr[i]:.4f}",
                    "Target": f"{tgt_arr[i]:.4f}",
                    "|Diff|": f"{abs(src_arr[i] - tgt_arr[i]):.4f}",
                    "Product": f"{src_arr[i] * tgt_arr[i]:.4f}",
                }
        st.dataframe(pd.DataFrame(pair_data).T, use_container_width=True)
    else:
        st.warning("Feature vectors not available.")
