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
from app.theme import inject_css

st.set_page_config(
    page_title="FlyMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# --- Sidebar ---
st.sidebar.title("🧠 FlyMind")
st.sidebar.caption("FlyWire Connectome Research")

# Demo mode
demo_mode = st.sidebar.checkbox("Demo Mode", value=False, help="Load verified example neurons")
demo_neuron = None
if demo_mode:
    demo_option = st.sidebar.selectbox(
        "Example neuron",
        [
            ("720575940597856265 — Tm16 (ACH, optic)"),
            ("720575940602380768 — CB1538 (GABA, central)"),
            ("720575940609282825 — lLN2F_b (GABA, central)"),
        ],
        key="demo_neuron_select",
    )
    demo_neuron = int(demo_option.split(" —")[0])

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "ML Pipeline",
        "Neuron Explorer",
        "Connection Predictor",
        "Candidate Ranking",
        "Research Results",
        "About / Methodology",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<small style="color:#6B7180">Candidate rankings are model-suggested '
    'hypotheses, not confirmed biological discoveries.</small>',
    unsafe_allow_html=True,
)

# --- Page routing ---
if page == "Overview":
    from app.pages import overview
    overview.render()
elif page == "ML Pipeline":
    from app.pages import pipeline_overview
    pipeline_overview.render()
elif page == "Neuron Explorer":
    from app.pages import neuron_explorer
    neuron_explorer.render(demo_id=demo_neuron)
elif page == "Connection Predictor":
    from app.pages import connection_predictor
    connection_predictor.render()
elif page == "Candidate Ranking":
    from app.pages import candidate_ranking
    candidate_ranking.render(demo_id=demo_neuron)
elif page == "Research Results":
    from app.pages import research_results
    research_results.render()
elif page == "About / Methodology":
    from app.pages import about
    about.render()
