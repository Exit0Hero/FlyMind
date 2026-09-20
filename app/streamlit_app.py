"""FlyMind Interactive Research Dashboard.

Entry point for the Streamlit application.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
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

# --- Custom CSS for scientific look ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; border: 1px solid #e9ecef; }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 0.3rem; }
    h3 { color: #0f3460; }
    .disclaimer { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 0.9em; }
    .observed { color: #28a745; font-weight: bold; }
    .candidate { color: #fd7e14; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- Sidebar navigation ---
st.sidebar.title("FlyMind")
st.sidebar.markdown("*FlyWire Connectome Research*")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
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
    "**Disclaimer:** Candidate rankings represent model-suggested "
    "connection hypotheses, not confirmed biological discoveries."
)


# --- Page routing ---
if page == "Overview":
    from app.pages import overview
    overview.render()
elif page == "Neuron Explorer":
    from app.pages import neuron_explorer
    neuron_explorer.render()
elif page == "Connection Predictor":
    from app.pages import connection_predictor
    connection_predictor.render()
elif page == "Candidate Ranking":
    from app.pages import candidate_ranking
    candidate_ranking.render()
elif page == "Research Results":
    from app.pages import research_results
    research_results.render()
elif page == "About / Limitations":
    from app.pages import about
    about.render()
