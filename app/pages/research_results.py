"""Research Results page — validated experiment results and evaluation metrics."""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
from app.services import get_store
from app.theme import inject_css, render_section_label, render_badge, render_disclaimer

RESULTS_DIR = Path("results/reports")

# Chart style constants
COLORS = {
    "primary": "#4FD1C5",
    "secondary": "#8B7CF6",
    "success": "#3FBE8C",
    "warning": "#E3B341",
    "error": "#E5534B",
    "text": "#EDEFF4",
    "text_muted": "#6B7180",
    "grid": "#232A38",
    "bg": "#12161F",
}


def _load_json(name):
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _setup_chart_style():
    """Apply consistent chart styling."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text_muted"],
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "text.color": COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def render():
    inject_css()
    _setup_chart_style()
    store = get_store()

    st.markdown("# Research Results")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Validated experiment results from the FlyMind connectome research.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Link Prediction", "Cold-Start", "Robustness", "Feature Importance", "Biological Patterns"
    ])

    # Tab 1: Link Prediction
    with tab1:
        st.markdown("## Experiment 2A — Random Edge Evaluation")
        st.markdown('<p style="color:#A6ADBB; font-size:14px">Setup: Randomly sample pairs, score them, evaluate ranking quality.</p>', unsafe_allow_html=True)

        results = _load_json("link_prediction_results.json")
        if results:
            rf = results.get("random_forest", {})
            if rf:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("ROC-AUC", f"{rf.get('roc_auc', 0):.4f}")
                mc2.metric("PR-AUC", f"{rf.get('pr_auc', 0):.4f}")
                mc3.metric("F1", f"{rf.get('f1', 0):.4f}")
                mc4.metric("Accuracy", f"{rf.get('accuracy', 0):.4f}")
        else:
            st.info("Random-edge results not found.")

        st.markdown("---")
        st.markdown("## Experiment 2B — Candidate Ranking")
        exp2d = _load_json("experiment_2d_results.json")
        if exp2d:
            metrics = exp2d.get("metrics", {})
            if metrics:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Recall@10", f"{metrics.get('recall_at_10', 0):.4f}")
                rc2.metric("Recall@50", f"{metrics.get('recall_at_50', 0):.4f}")
                rc3.metric("Hit Rate@10", f"{metrics.get('hit_rate_at_10', 0):.4f}")
                rc4.metric("Hit Rate@50", f"{metrics.get('hit_rate_at_50', 0):.4f}")
        else:
            st.info("Experiment 2D results not found.")

    # Tab 2: Cold-Start
    with tab2:
        st.markdown("## Cold-Start Generalization")
        st.markdown("""
        <div style="background:rgba(79,209,197,0.08); border:1px solid rgba(79,209,197,0.2); border-radius:10px; padding:16px; margin-bottom:16px">
            <strong style="color:#4FD1C5">Cold-start evaluation</strong>
            <span style="color:#A6ADBB"> tests whether the model generalizes to previously unseen neurons.
            Training edges touching held-out neurons are excluded.</span>
        </div>
        """, unsafe_allow_html=True)

        cs = _load_json("link_prediction_cold_start_results.json")
        if cs:
            rf = cs.get("random_forest", {})
            if rf:
                st.markdown("### Random Forest (Cold-Start)")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("ROC-AUC", f"{rf.get('roc_auc', 0):.4f}")
                sc2.metric("PR-AUC", f"{rf.get('pr_auc', 0):.4f}")
                sc3.metric("F1", f"{rf.get('f1', 0):.4f}")
                sc4.metric("Accuracy", f"{rf.get('accuracy', 0):.4f}")

            gnn = cs.get("graphsage", {})
            if gnn:
                st.markdown("### GraphSAGE (Cold-Start)")
                st.markdown('<p style="color:#A6ADBB">GraphSAGE was evaluated as a comparative model. The validated RF model was stronger.</p>', unsafe_allow_html=True)
                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric("ROC-AUC", f"{gnn.get('roc_auc', 0):.4f}")
                gc2.metric("PR-AUC", f"{gnn.get('pr_auc', 0):.4f}")
                gc3.metric("F1", f"{gnn.get('f1', 0):.4f}")
                gc4.metric("Accuracy", f"{gnn.get('accuracy', 0):.4f}")
        else:
            st.info("Cold-start results not found.")

    # Tab 3: Robustness
    with tab3:
        st.markdown("## Experiment 3 — Robustness and Calibration")
        robust = _load_json("experiment_3_robustness.json")
        if robust:
            ps = robust.get("3a_per_source_ranking", {})
            if ps:
                st.markdown("### Per-Source Ranking")
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Recall@10", f"{ps.get('mean_recall@10', 0):.4f}")
                rc2.metric("Recall@50", f"{ps.get('mean_recall@50', 0):.4f}")
                rc3.metric("Hit Rate@10", f"{ps.get('hit_rate@10', 0):.4f}")
                rc4.metric("Hit Rate@50", f"{ps.get('hit_rate@50', 0):.4f}")

            cal = robust.get("3d_calibration", {})
            if cal:
                st.markdown("### Calibration")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Brier Score", f"{cal.get('brier_score', 0):.4f}")
                cc2.metric("Log Loss", f"{cal.get('log_loss', 0):.4f}")
                cc3.metric("ECE", f"{cal.get('ece', 0):.4f}")

            # Calibration curve
            cal_data = store.get_calibration_data()
            if cal_data:
                predicted = [c.get("mean_predicted", 0) for c in cal_data]
                observed = [c.get("observed_positive_rate", 0) for c in cal_data]
                if predicted and observed:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot([0, 1], [0, 1], "--", color=COLORS["text_muted"], label="Perfect calibration", alpha=0.5)
                    ax.plot(predicted, observed, "o-", color=COLORS["primary"], label="FlyMind RF", markersize=6)
                    ax.fill_between(predicted, observed, alpha=0.1, color=COLORS["primary"])
                    ax.set_xlabel("Mean predicted probability")
                    ax.set_ylabel("Observed frequency")
                    ax.set_title("Calibration Curve")
                    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("Experiment 3 results not found.")

    # Tab 4: Feature Importance
    with tab4:
        st.markdown("## Feature Importance")
        st.markdown(render_disclaimer("Feature importance describes model behavior; it does not establish biological causation."), unsafe_allow_html=True)

        fi = store.get_feature_importance()
        if fi:
            fi_df = pd.DataFrame([{"feature": f.feature, "importance": f.importance, "group": f.group} for f in fi])

            fig, ax = plt.subplots(figsize=(10, 6))
            top = fi_df.head(15)
            color_map = {"neurotransmitter": COLORS["primary"], "morphology": COLORS["success"],
                         "spatial": COLORS["secondary"], "classification": COLORS["warning"]}
            colors = [color_map.get(g, COLORS["text_muted"]) for g in top["group"]]
            bars = ax.barh(range(len(top)), top["importance"].values, color=colors, height=0.6)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"].values)
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top 15 Feature Importances")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLORS["primary"], label="Neurotransmitter"),
                Patch(facecolor=COLORS["success"], label="Morphology"),
                Patch(facecolor=COLORS["secondary"], label="Spatial"),
                Patch(facecolor=COLORS["warning"], label="Classification"),
            ]
            st.markdown("""
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px">
                <span style="font-size:12px; color:#4FD1C5">● Neurotransmitter</span>
                <span style="font-size:12px; color:#3FBE8C">● Morphology</span>
                <span style="font-size:12px; color:#8B7CF6">● Spatial</span>
                <span style="font-size:12px; color:#E3B341">● Classification</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Feature importance data not available.")

    # Tab 5: Biological Patterns
    with tab5:
        st.markdown("## Biological Pattern Analysis")
        st.markdown('<p style="color:#A6ADBB">Post-hoc analysis of model predictions by biological properties. These are descriptive statistics, not causal claims.</p>', unsafe_allow_html=True)

        cal_data = store.get_calibration_data()
        if cal_data:
            predicted = [c.get("mean_predicted", 0) for c in cal_data]
            observed = [c.get("observed_positive_rate", 0) for c in cal_data]
            if predicted and observed:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot([0, 1], [0, 1], "--", color=COLORS["text_muted"], label="Perfect calibration", alpha=0.5)
                ax.plot(predicted, observed, "o-", color=COLORS["primary"], label="FlyMind RF", markersize=6)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_title("Calibration Curve")
                ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("### Prediction by Super-class")
        st.markdown('<p style="color:#A6ADBB">Post-hoc analysis shows the model\'s ranking behavior varies by neuron super-class. This reflects dataset composition and biological heterogeneity, not model failure.</p>', unsafe_allow_html=True)

        st.markdown(render_disclaimer("The model does not explicitly model biological class membership. Super-class differences emerge from feature distributions."), unsafe_allow_html=True)
