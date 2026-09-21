"""Overview page — project summary and key metrics."""

import streamlit as st
from app.theme import inject_css, render_metric_card, render_badge, render_card, render_disclaimer, render_section_label, render_display_number

def render():
    inject_css()

    # Hero
    st.markdown("""
    <div style="text-align:center; padding: 48px 0 32px">
        <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#6B7180; margin-bottom:12px">
            FLYWIRE CONNECTOME RESEARCH
        </div>
        <h1 style="font-family:'Inter Tight',sans-serif; font-weight:700; font-size:48px; letter-spacing:-0.02em; color:#EDEFF4; margin:0 0 12px">
            FlyMind
        </h1>
        <p style="font-size:16px; color:#A6ADBB; max-width:600px; margin:0 auto; line-height:1.6">
            Learning neuron connectivity from biological and morphological properties
            in the Drosophila melanogaster connectome.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    st.markdown(render_section_label("KEY METRICS"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Neurons", "139,255")
    with c2:
        st.metric("Directed Edges", "3.73M")
    with c3:
        st.metric("Cold-Start ROC-AUC", "0.9800")
    with c4:
        st.metric("Recall@10", "0.814")

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Pipeline strip
    st.markdown(render_section_label("ML PIPELINE"), unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0; overflow-x:auto; padding:8px 0">
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">📊</div>
            <div class="node-label">Data</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">⚙️</div>
            <div class="node-label">ETL</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">🔬</div>
            <div class="node-label">Features</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">🤖</div>
            <div class="node-label">Model</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">📈</div>
            <div class="node-label">Evaluation</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">🎯</div>
            <div class="node-label">Prediction</div>
            <span class="node-status complete"></span>
        </div>
        <div class="pipeline-connector">→</div>
        <div class="pipeline-node" style="flex:0 0 auto">
            <div class="node-icon">🏆</div>
            <div class="node-label">Ranking</div>
            <span class="node-status complete"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Quick actions
    st.markdown(render_section_label("QUICK ACTIONS"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("app/streamlit_app.py", label="🔍 Explore a Neuron", icon=None)
    with c2:
        st.page_link("app/streamlit_app.py", label="🎯 Predict a Connection", icon=None)
    with c3:
        st.page_link("app/streamlit_app.py", label="🏆 View Top Candidates", icon=None)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Research question
    st.markdown(render_section_label("RESEARCH QUESTION"), unsafe_allow_html=True)
    st.markdown("""
    <div class="flymind-card">
        <p style="font-size:16px; color:#EDEFF4; line-height:1.6; margin:0">
            Can neuron-level biological and morphological properties predict directed
            connectivity in the fruit-fly connectome, and do those relationships
            generalize to previously unseen neurons?
        </p>
        <p style="font-size:14px; color:#3FBE8C; margin:12px 0 0; font-weight:600">
            ✓ Yes — the RF model achieves ROC-AUC 0.980 on cold-start evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(render_disclaimer(), unsafe_allow_html=True)
