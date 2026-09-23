"""Candidate Ranking page — rank candidate target connections for a source neuron."""

import streamlit as st
import pandas as pd
from app.services import get_store
from app.theme import inject_css, render_section_label, render_badge, render_score_bar, render_disclaimer, render_monospace

def render(demo_id=None):
    inject_css()
    store = get_store()
    df = store.neuron_table

    st.markdown("# Candidate Ranking")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Rank candidate target neurons for a given source neuron based on model scoring.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Source Selection
    st.markdown(render_section_label("SOURCE NEURON"), unsafe_allow_html=True)
    source_id = st.text_input(
        "Source neuron root_id",
        value=str(demo_id) if demo_id else "",
        placeholder="e.g. 720575940597856265",
        label_visibility="collapsed",
    )

    if not source_id:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#6B7180">
            <div style="font-size:48px; margin-bottom:16px">🏆</div>
            <div style="font-size:16px">Enter a source neuron ID to rank candidate targets</div>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        src = int(source_id)
    except ValueError:
        st.error("Invalid root_id. Must be an integer.")
        return

    src_data = store.get_neuron(src)
    if src_data is None:
        st.error(f"Neuron {src} not found.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of candidates (K)", 5, 100, 20)
    with col2:
        exclude_observed = st.checkbox("Exclude observed connections", value=True)

    # Rank candidates
    with st.spinner("Ranking candidates..."):
        results = store.rank_candidate_targets(src, k=k, exclude_observed=exclude_observed)

    if not results:
        st.warning("No candidates found.")
        return

    # Source info
    st.markdown(f"""
    <div class="flymind-card" style="margin-bottom:16px">
        <div style="display:flex; align-items:center; gap:12px">
            <div>
                <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180; margin-bottom:4px">
                    SOURCE
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#4FD1C5">
                    {src}
                </div>
            </div>
            <div style="font-size:14px; color:#A6ADBB">
                {src_data.primary_type or '—'} ({src_data.super_class or '—'})
            </div>
            <div style="margin-left:auto">
                {render_badge(src_data.nt_type or 'Unknown', 'teal' if src_data.nt_type else 'muted')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown(render_disclaimer(), unsafe_allow_html=True)

    st.markdown("---")

    # Results
    st.markdown(render_section_label(f"TOP {len(results)} CANDIDATE CONNECTIONS"), unsafe_allow_html=True)

    # Summary metrics
    if results:
        scores = [r.score for r in results]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Top Score", f"{max(scores):.4f}")
        mc2.metric("Median Score", f"{scores[len(scores)//2]:.4f}")
        mc3.metric("Score Range", f"{min(scores):.4f} – {max(scores):.4f}")

    # Candidate table with score bars
    for r in results:
        tgt_data = store.get_neuron(r.target_root_id)
        is_observed = store.check_connection(src, r.target_root_id)

        st.markdown(f"""
        <div class="flymind-card" style="margin-bottom:8px; padding:16px">
            <div style="display:flex; align-items:center; gap:16px">
                <div style="min-width:40px; text-align:center">
                    <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180">RANK</div>
                    <div style="font-family:'Inter Tight',sans-serif; font-weight:700; font-size:24px; color:#EDEFF4">{r.rank}</div>
                </div>
                <div style="flex:1">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#8B7CF6">{r.target_root_id}</span>
                        {render_badge('PREDICTED', 'violet')}
                        {render_badge('OBSERVED', 'success') if is_observed else ''}
                    </div>
                    <div style="font-size:13px; color:#A6ADBB">
                        {tgt_data.primary_type or '—'} · {tgt_data.super_class or '—'} · {tgt_data.nt_type or 'Unknown NT'}
                    </div>
                </div>
                <div style="min-width:200px">
                    <div style="font-family:'Inter Tight',sans-serif; font-weight:700; font-size:20px; color:#8B7CF6; text-align:right; margin-bottom:4px">
                        {r.score:.4f}
                    </div>
                    {render_score_bar(r.score, 'predicted')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
