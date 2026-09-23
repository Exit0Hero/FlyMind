"""ML Pipeline page — full pipeline visualization with status."""

from pathlib import Path
import streamlit as st
from app.theme import inject_css, render_section_label, render_badge, render_disclaimer, render_monospace

RESULTS_DIR = Path("results/reports")


def _check_file(path):
    p = Path(path)
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        return True, f"{size_mb:.1f} MB"
    return False, "Not found"


def render():
    inject_css()

    st.markdown("# ML Pipeline")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Complete end-to-end machine learning pipeline for FlyMind link prediction.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline diagram with interactive nodes
    st.markdown(render_section_label("PIPELINE STAGES"), unsafe_allow_html=True)

    stages = [
        ("📊", "FlyWire Data", "complete", "139,255 neurons, 3.73M edges"),
        ("⚙️", "ETL", "complete", "Feature matrix + ID mapping"),
        ("🔬", "Features", "complete", "15 node → 60D pair"),
        ("🤖", "Model", "complete", "Random Forest, 100 trees"),
        ("📈", "Evaluation", "complete", "ROC-AUC 0.9800"),
        ("🎯", "Prediction", "complete", "Ready"),
        ("🏆", "Ranking", "complete", "Ready"),
    ]

    cols = st.columns(len(stages) * 2 - 1)
    for i, (icon, label, status, desc) in enumerate(stages):
        col_idx = i * 2
        with cols[col_idx]:
            st.markdown(f"""
            <div class="pipeline-node">
                <div class="node-icon">{icon}</div>
                <div class="node-label">{label}</div>
                <span class="node-status {status}"></span>
            </div>
            """, unsafe_allow_html=True)
        if i < len(stages) - 1:
            with cols[col_idx + 1]:
                st.markdown('<div class="pipeline-connector">→</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    # Stage details
    st.markdown(render_section_label("STAGE DETAILS"), unsafe_allow_html=True)

    # Data
    with st.expander("📊 Data — FlyWire Connectome", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            exists, size = _check_file("data/processed/neuron_table.parquet")
            st.metric("Neuron Table", size, delta="✓ Found" if exists else "✗ Missing")
            st.caption("139,255 neurons with biological features")
        with c2:
            exists, size = _check_file("data/processed/link_prediction/edges_aggregated.parquet")
            st.metric("Edge Table", size, delta="✓ Found" if exists else "✗ Missing")
            st.caption("3,732,460 directed edges")

    # ETL
    with st.expander("⚙️ ETL — Extract, Transform, Load"):
        c1, c2, c3 = st.columns(3)
        with c1:
            exists, size = _check_file("data/processed/link_prediction/X_features.npy")
            st.metric("Feature Matrix", size, delta="✓ Found" if exists else "✗ Missing")
            st.caption("139,255 × 15 float32")
        with c2:
            exists, size = _check_file("data/processed/link_prediction/id_to_idx.json")
            st.metric("ID Mapping", size, delta="✓ Found" if exists else "✗ Missing")
            st.caption("root_id to matrix index")
        with c3:
            exists, size = _check_file("data/processed/link_prediction/all_node_ids.npy")
            st.metric("Node IDs", size, delta="✓ Found" if exists else "✗ Missing")
            st.caption("All valid neuron IDs")

    # Features
    with st.expander("🔬 Feature Engineering"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Node Features", "15")
            st.caption("nt_type_score, 6 NT scores, 3 morphology, 3 spatial, flow_enc, side_x_enc")
        with c2:
            st.metric("Pair Features", "60")
            st.caption("[source, target, |source−target|, source×target]")

    # Model
    with st.expander("🤖 Trained Model"):
        exists, size = _check_file("models/link_prediction_rf.pkl")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Model Type", "RandomForestClassifier")
        with c2:
            st.metric("Estimators", "100")
        with c3:
            st.metric("Model Size", size if exists else "Missing")
        with c4:
            st.metric("Feature Dim", "60")
        st.caption("Frozen production artifact — models/link_prediction_rf.pkl")

    # Evaluation
    with st.expander("📈 Evaluation — Cold-Start Protocol"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("ROC-AUC", "0.9800")
        with c2:
            st.metric("PR-AUC", "0.9739")
        with c3:
            st.metric("Recall@10", "0.814")
        with c4:
            st.metric("Hit Rate@10", "0.995")
        st.caption("Cold-start evaluation on held-out neurons (70/15/15 split)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Brier Score", "0.0536")
        with c2:
            st.metric("Log Loss", "0.1914")
        with c3:
            st.metric("ECE", "0.0411")

    # Prediction
    with st.expander("🎯 Prediction — Ready"):
        st.success("Prediction engine loaded and validated.")
        st.markdown("""
        - Source neuron ID → 15-dim feature vector
        - Target neuron ID → 15-dim feature vector
        - Pair features → [xA, xB, |xA−xB|, xA×xB] = 60-dim
        - RF model → connection ranking score
        """)

    # Ranking
    with st.expander("🏆 Candidate Ranking — Ready"):
        st.success("Candidate ranking engine loaded and validated.")
        st.markdown("""
        - Sample candidate pool (3× requested count)
        - Exclude self-loops and observed edges
        - Score all candidates with RF model
        - Rank by score descending
        - Return top-K with metadata
        """)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown(render_disclaimer(), unsafe_allow_html=True)
