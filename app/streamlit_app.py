"""FlyMind Interactive Research Dashboard.

Entry point for the Streamlit application.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="FlyMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for scientific/research look ---
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1 { color: #1a1a2e; font-size: 2rem; }
    h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 0.3rem; font-size: 1.4rem; }
    h3 { color: #0f3460; font-size: 1.15rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; border: 1px solid #e9ecef; }
    .disclaimer { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 0.9em; }
    .disclaimer-info { background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 0.9em; }
    .observed { color: #28a745; font-weight: bold; }
    .candidate { color: #fd7e14; font-weight: bold; }
    .hero-text { font-size: 1.1rem; color: #495057; margin-bottom: 1rem; }
    .metric-highlight { font-size: 2rem; font-weight: bold; color: #0f3460; }
    .section-divider { border-top: 1px solid #dee2e6; margin: 1.5rem 0; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# --- Demo mode in sidebar ---
st.sidebar.title("FlyMind")
st.sidebar.markdown("*FlyWire Connectome Research*")

# Demo mode toggle
demo_mode = st.sidebar.checkbox("Demo Mode", value=False, help="Load verified example neurons for quick demo")

if demo_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Demo:**")
    demo_neuron = st.sidebar.selectbox(
        "Example neuron",
        [
            ("720575940597856265 — Tm16 (ACH, optic)"),
            ("720575940602380768 — CB1538 (GABA, central)"),
            ("720575940609282825 — lLN2F_b (GABA, central)"),
        ],
        key="demo_neuron_select",
    )
    st.sidebar.caption("Verified neurons with known annotations.")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Pipeline Overview",
        "Neuron Explorer",
        "Connection Predictor",
        "Candidate Ranking",
        "Research Results",
        "About / Limitations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>**Disclaimer:** Candidate rankings represent model-suggested "
    "connection hypotheses, not confirmed biological discoveries.</small>",
    unsafe_allow_html=True,
)

# --- Page routing ---
if page == "Overview":
    from app.pages import overview
    overview.render()
elif page == "Pipeline Overview":
    from app.pages import pipeline_overview
    pipeline_overview.render()
elif page == "Neuron Explorer":
    from app.pages import neuron_explorer
    neuron_explorer.render(demo_id=int(demo_neuron.split(" —")[0]) if demo_mode else None)
elif page == "Connection Predictor":
    from app.pages import connection_predictor
    connection_predictor.render()
elif page == "Candidate Ranking":
    from app.pages import candidate_ranking
    candidate_ranking.render(demo_id=int(demo_neuron.split(" —")[0]) if demo_mode else None)
elif page == "Research Results":
    from app.pages import research_results
    research_results.render()
elif page == "About / Limitations":
    from app.pages import about
    about.render()
