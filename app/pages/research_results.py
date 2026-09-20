"""Research Results page — validated experiment results and evaluation metrics."""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from app.services import get_store

RESULTS_DIR = Path("results/reports")


def _load_json(name):
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def render():
    store = get_store()

    st.markdown("# Research Results")
    st.markdown("Validated experiment results from the FlyMind connectome research.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Link Prediction", "Cold-Start", "Robustness", "Feature Importance", "Biological Patterns"
    ])

    # --- Tab 1: Link Prediction ---
    with tab1:
        st.markdown("## Experiment 2A — Random Edge Evaluation")
        st.markdown("""
        **Setup:** Randomly sample pairs, score them, evaluate ranking quality.

        - **Positive:** Observed directed connection
        - **Negative:** Candidate pair not present in the evaluated graph
        """)
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
        st.markdown("""
        For each source neuron, rank all non-observed targets.

        - **Recall@K:** Fraction of true connections in top-K ranked targets
        - **Hit Rate@K:** Fraction of queries with at least one true connection in top-K
        """)
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

    # --- Tab 2: Cold-Start ---
    with tab2:
        st.markdown("## Cold-Start Generalization")
        st.markdown("""
        <div class="disclaimer-info">
        <strong>Cold-start evaluation</strong> tests whether the model generalizes to
        <em>previously unseen neurons</em>. Training edges touching held-out neurons are excluded.
        The model is evaluated entirely on held-out neurons.
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
                st.markdown("""
                GraphSAGE was evaluated as a comparative model. The validated Random Forest
                model was stronger in the evaluated link-prediction experiments.
                """)
                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric("ROC-AUC", f"{gnn.get('roc_auc', 0):.4f}")
                gc2.metric("PR-AUC", f"{gnn.get('pr_auc', 0):.4f}")
                gc3.metric("F1", f"{gnn.get('f1', 0):.4f}")
                gc4.metric("Accuracy", f"{gnn.get('accuracy', 0):.4f}")

            st.markdown("""
            <div class="disclaimer">
            <strong>Conclusion:</strong> Node-level biological and morphological information
            generalized strongly to held-out neurons in this evaluation. This does not imply
            universal generalization to all possible neurons.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Cold-start results not found.")

    # --- Tab 3: Robustness ---
    with tab3:
        st.markdown("## Experiment 3 — Robustness and Calibration")
        robust = _load_json("experiment_3_robustness.json")
        if robust:
            # Per-source ranking
            st.markdown("### Per-Source Ranking")
            ps = robust.get("3a_per_source_ranking", {})
            if ps:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Recall@10", f"{ps.get('mean_recall@10', 0):.4f}")
                rc2.metric("Recall@50", f"{ps.get('mean_recall@50', 0):.4f}")
                rc3.metric("Hit Rate@10", f"{ps.get('hit_rate@10', 0):.4f}")
                rc4.metric("Hit Rate@50", f"{ps.get('hit_rate@50', 0):.4f}")

            # Calibration
            st.markdown("### Calibration")
            cal = robust.get("3d_calibration", {})
            if cal:
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Brier Score", f"{cal.get('brier_score', 0):.4f}")
                cc2.metric("Log Loss", f"{cal.get('log_loss', 0):.4f}")
                cc3.metric("ECE", f"{cal.get('ece', 0):.4f}")

            # Score saturation
            st.markdown("### Score Distribution")
            sat = robust.get("3c_score_saturation", {})
            if sat:
                s50 = sat.get("score_>=0.5", {})
                s90 = sat.get("score_>=0.9", {})
                s100 = sat.get("score_=1.0", {})
                st.markdown(f"Scores ≥ 0.50: {s50.get('percentage', 0):.1f}% ({s50.get('count', 0):,})")
                st.markdown(f"Scores ≥ 0.90: {s90.get('percentage', 0):.1f}% ({s90.get('count', 0):,})")
                st.markdown(f"Scores = 1.00: {s100.get('percentage', 0):.1f}% ({s100.get('count', 0):,})")

            # Held-out ranking
            st.markdown("### Held-Out Ranking")
            ho = robust.get("3f_rank_distribution", {})
            if ho:
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Median Rank", f"{ho.get('median_rank', 0):.0f}")
                hc2.metric("Top-1 Fraction", f"{ho.get('frac_ranked_1', 0):.4f}")
                hc3.metric("Top-10 Fraction", f"{ho.get('frac_in_top_10', 0):.4f}")
                hc4.metric("Top-50 Fraction", f"{ho.get('frac_in_top_50', 0):.4f}")

            # Exclusion audit
            st.markdown("### Exclusion Audit")
            ex = robust.get("3g_exclusion_audit", {})
            if ex:
                ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                ec1.metric("Self-loops", ex.get("self_loops", 0))
                ec2.metric("Known edges", ex.get("known_edges", 0))
                ec3.metric("Duplicates", ex.get("duplicate_pairs", 0))
                ec4.metric("Invalid IDs", ex.get("invalid_ids", 0))
                ec5.metric("Non-finite scores", ex.get("non_finite_scores", 0))
        else:
            st.info("Experiment 3 results not found.")

    # --- Tab 4: Feature Importance ---
    with tab4:
        st.markdown("## Feature Importance")
        st.markdown("""
        <div class="disclaimer">
        <strong>Note:</strong> Feature importance describes model behavior; it does not
        establish biological causation.
        </div>
        """, unsafe_allow_html=True)

        fi = store.get_feature_importance()
        if fi:
            fi_df = pd.DataFrame(fi)
            # Group by feature type
            source_features = fi_df[fi_df["feature"].str.startswith("src_")]
            target_features = fi_df[fi_df["feature"].str.startswith("tgt_")]
            diff_features = fi_df[fi_df["feature"].str.startswith("diff_")]
            product_features = fi_df[fi_df["feature"].str.startswith("product_")]
            base_features = fi_df[~fi_df["feature"].str.startswith(("src_", "tgt_", "diff_", "product_"))]

            fig, ax = plt.subplots(figsize=(10, 8))
            top = fi_df.head(20)
            colors = []
            for f in top["feature"]:
                if f.startswith("src_"):
                    colors.append("#2196F3")
                elif f.startswith("tgt_"):
                    colors.append("#4CAF50")
                elif f.startswith("diff_"):
                    colors.append("#FF9800")
                elif f.startswith("product_"):
                    colors.append("#9C27B0")
                else:
                    colors.append("#607D8B")
            ax.barh(range(len(top)), top["importance"].values, color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"].values)
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Feature Importances")
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2196F3", label="Source biology"),
                Patch(facecolor="#4CAF50", label="Target biology"),
                Patch(facecolor="#FF9800", label="Difference"),
                Patch(facecolor="#9C27B0", label="Product (interaction)"),
                Patch(facecolor="#607D8B", label="Base pair"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            **Feature categories:**
            - <span style='color:#2196F3'>● Source biology</span> — source neuron features
            - <span style='color:#4CAF50'>● Target biology</span> — target neuron features
            - <span style='color:#FF9800'>● Difference</span> — |source - target|
            - <span style='color:#9C27B0'>● Product</span> — source × target interaction
            - <span style='color:#607D8B'>● Base pair</span> — original 15 features
            """, unsafe_allow_html=True)
        else:
            st.info("Feature importance data not available.")

    # --- Tab 5: Biological Patterns ---
    with tab5:
        st.markdown("## Biological Pattern Analysis")
        st.markdown("""
        Post-hoc analysis of model predictions by biological properties.
        These are descriptive statistics, not causal claims.
        """)

        # Show calibration data
        cal_data = store.get_calibration_data()
        if cal_data:
            predicted = [c.get("mean_predicted", 0) for c in cal_data]
            observed = [c.get("observed_positive_rate", 0) for c in cal_data]

            if predicted and observed:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
                ax.plot(predicted, observed, "bo-", label="FlyMind RF")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_title("Calibration Curve")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Super-class analysis
        st.markdown("### Prediction by Super-class")
        st.markdown("""
        Post-hoc analysis shows the model's ranking behavior varies by neuron super-class.
        This reflects dataset composition and biological heterogeneity, not model failure.
        """)

        st.markdown("""
        <div class="disclaimer">
        <strong>Limitation:</strong> The model does not explicitly model biological class membership.
        Super-class differences emerge from feature distributions, not from supervised class signals.
        </div>
        """, unsafe_allow_html=True)
