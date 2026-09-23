"""FlyMind Dashboard — Cached data services.

Provides Streamlit-cached access to the FlyMindDataStore singleton.
All data access goes through this module to avoid redundant loads.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.link_prediction.presentation import FlyMindDataStore, get_datastore


@st.cache_resource
def get_store() -> FlyMindDataStore:
    """Return the cached FlyMindDataStore singleton.

    Uses Streamlit's cache_resource so the RF model, edge indices,
    and neuron table are loaded exactly once per server session.
    """
    return get_datastore()
