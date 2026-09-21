"""FlyMind Design System — Central CSS injection and design tokens.

All design tokens and CSS are injected once via inject_css().
Pages use the helper functions for consistent styling.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg_base": "#0B0E14",
    "bg_surface": "#12161F",
    "bg_elevated": "#1A1F2B",
    "border_subtle": "#232A38",
    "border_strong": "#333C4F",
    "text_primary": "#EDEFF4",
    "text_secondary": "#A6ADBB",
    "text_muted": "#6B7180",
    "accent_primary": "#4FD1C5",
    "accent_secondary": "#8B7CF6",
    "success": "#3FBE8C",
    "warning": "#E3B341",
    "error": "#E5534B",
}

# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #0B0E14;
    --bg-surface: #12161F;
    --bg-elevated: #1A1F2B;
    --border-subtle: #232A38;
    --border-strong: #333C4F;
    --text-primary: #EDEFF4;
    --text-secondary: #A6ADBB;
    --text-muted: #6B7180;
    --accent-primary: #4FD1C5;
    --accent-secondary: #8B7CF6;
    --success: #3FBE8C;
    --warning: #E3B341;
    --error: #E5534B;
}

/* Base typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Headings */
h1 { font-family: 'Inter Tight', 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h2 { font-weight: 600 !important; letter-spacing: -0.01em !important; }
h3 { font-weight: 600 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 120ms ease-out !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: var(--bg-elevated) !important;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    color: var(--accent-primary) !important;
    border-left: 3px solid var(--accent-primary) !important;
    padding-left: 9px !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    transition: all 120ms ease-out !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.28) !important;
    border-color: var(--border-strong) !important;
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Inter Tight', 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 32px !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 13px !important;
}

/* DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--text-secondary) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-primary) !important;
    border-bottom-color: var(--accent-primary) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 80ms ease-out !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: var(--accent-primary) !important;
    color: var(--bg-base) !important;
    border: none !important;
}

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    background: var(--bg-surface) !important;
}

/* Info/Warning/Error boxes */
div[data-testid="stInfo"] {
    background: rgba(79, 209, 197, 0.08) !important;
    border: 1px solid rgba(79, 209, 197, 0.2) !important;
    border-radius: 10px !important;
}
div[data-testid="stWarning"] {
    background: rgba(227, 179, 65, 0.08) !important;
    border: 1px solid rgba(227, 179, 65, 0.2) !important;
    border-radius: 10px !important;
}
div[data-testid="stError"] {
    background: rgba(229, 83, 75, 0.08) !important;
    border: 1px solid rgba(229, 83, 75, 0.2) !important;
    border-radius: 10px !important;
}
div[data-testid="stSuccess"] {
    background: rgba(63, 190, 140, 0.08) !important;
    border: 1px solid rgba(63, 190, 140, 0.2) !important;
    border-radius: 10px !important;
}

/* Score bar component */
.score-bar-container {
    background: var(--bg-elevated);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 300ms ease-out;
}
.score-bar-fill.observed { background: var(--accent-primary); }
.score-bar-fill.predicted { background: var(--accent-secondary); }

/* Card component */
.flymind-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 20px;
    transition: all 120ms ease-out;
}
.flymind-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.28);
    border-color: var(--border-strong);
}

/* Badge component */
.flymind-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.flymind-badge.teal { background: rgba(79,209,197,0.15); color: #4FD1C5; }
.flymind-badge.violet { background: rgba(139,124,246,0.15); color: #8B7CF6; }
.flymind-badge.success { background: rgba(63,190,140,0.15); color: #3FBE8C; }
.flymind-badge.warning { background: rgba(227,179,65,0.15); color: #E3B341; }
.flymind-badge.error { background: rgba(229,83,75,0.15); color: #E5534B; }
.flymind-badge.muted { background: rgba(107,113,128,0.15); color: #6B7180; }

/* Pipeline node */
.pipeline-node {
    background: var(--bg-surface);
    border: 2px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 200ms cubic-bezier(0.16,1,0.3,1);
    cursor: pointer;
    min-width: 120px;
}
.pipeline-node:hover {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(79,209,197,0.15);
    transform: translateY(-2px);
}
.pipeline-node.active {
    border-color: var(--accent-primary);
    background: var(--bg-elevated);
}
.pipeline-node .node-icon { font-size: 24px; margin-bottom: 6px; }
.pipeline-node .node-label { font-size: 12px; font-weight: 600; color: var(--text-secondary); }
.pipeline-node .node-status {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-top: 6px;
}
.pipeline-node .node-status.complete { background: var(--success); }
.pipeline-node .node-status.error { background: var(--error); }
.pipeline-node .node-status.pending { background: var(--text-muted); }

/* Pipeline connector */
.pipeline-connector {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--border-strong);
    font-size: 20px;
    min-width: 40px;
}

/* Monospace text */
.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* Display number */
.display-number {
    font-family: 'Inter Tight', 'Inter', sans-serif;
    font-weight: 700;
    font-size: 48px;
    letter-spacing: -0.02em;
    line-height: 1.0;
    color: var(--text-primary);
}

/* Section label */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* Disclaimer */
.flymind-disclaimer {
    background: rgba(227,179,65,0.08);
    border: 1px solid rgba(227,179,65,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Fade-in animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 200ms ease-out forwards; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Main container */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
"""


def inject_css():
    """Inject the complete FlyMind design system CSS. Call once per page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable Component Helpers
# ---------------------------------------------------------------------------
def render_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_badge(text: str, color: str = "teal"):
    """Render an inline badge. Colors: teal, violet, success, warning, error, muted."""
    return f'<span class="flymind-badge {color}">{text}</span>'


def render_score_bar(score: float, kind: str = "predicted", width_pct: float = None):
    """Render a horizontal score bar. kind: 'observed' or 'predicted'."""
    pct = width_pct if width_pct is not None else score * 100
    color_class = "observed" if kind == "observed" else "predicted"
    return f"""
    <div class="score-bar-container">
        <div class="score-bar-fill {color_class}" style="width: {pct:.1f}%"></div>
    </div>
    """


def render_card(content: str, extra_class: str = ""):
    """Wrap content in a styled card."""
    return f'<div class="flymind-card {extra_class}">{content}</div>'


def render_pipeline_node(label: str, icon: str, status: str = "complete", active: bool = False):
    """Render a pipeline stage node. status: complete, error, pending."""
    active_class = " active" if active else ""
    return f"""
    <div class="pipeline-node{active_class}">
        <div class="node-icon">{icon}</div>
        <div class="node-label">{label}</div>
        <span class="node-status {status}"></span>
    </div>
    """


def render_pipeline_connector():
    """Render a pipeline connector arrow."""
    return '<div class="pipeline-connector">→</div>'


def render_disclaimer(text: str = None):
    """Render the scientific disclaimer box."""
    if text is None:
        text = ("Model-suggested candidate connections are hypotheses for further "
                "investigation and are not experimentally validated biological connections.")
    return f'<div class="flymind-disclaimer">{text}</div>'


def render_section_label(text: str):
    """Render an uppercase section label."""
    return f'<div class="section-label">{text}</div>'


def render_display_number(value: str, label: str = ""):
    """Render a large display number."""
    html = f'<div class="display-number">{value}</div>'
    if label:
        html += f'<div class="section-label" style="margin-top:4px">{label}</div>'
    return html


def render_monospace(text: str):
    """Render monospace text."""
    return f'<span class="mono">{text}</span>'
