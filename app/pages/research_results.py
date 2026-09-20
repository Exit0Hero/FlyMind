"""FlyMind Dashboard — Research Results Page."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render():
    st.title("Research Results")
    st.markdown("Verified experiment results from the FlyMind research phase.")

    store = get_store()
    results = store.get_experiment_results()

    # --- Tabs for experiments ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Link Prediction",
        "Cold-Start",
        "Robustness",
        "Feature Importance",
    ])

    # --- Tab 1: Random Edge Holdout (2A) ---
    with tab1:
        st.markdown("## Experiment 2A — Random Edge Holdout")
        st.markdown(
            "Standard train/val/test split. **Caution:** Nodes remain visible "
            "through other edges, making this an optimistic evaluation."
        )

        lp = results.get("2a", {})
        if lp:
            # Build comparison table
            rows = []
            model_map = {
                "rf_node_only": "RF Node Features",
                "rf_node_plus_heuristic": "RF + Heuristics",
                "gnn": "GraphSAGE",
                "random": "Random Baseline",
            }
            for key, display in model_map.items():
                if key in lp:
                    m = lp[key]
                    rows.append({
                        "Model": display,
                        "ROC-AUC": m.get("roc_auc"),
                        "PR-AUC": m.get("pr_auc"),
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
                st.bar_chart(df.set_index("Model")[["ROC-AUC", "PR-AUC"]])

    # --- Tab 2: Cold Start (2B) ---
    with tab2:
        st.markdown("## Experiment 2B — Cold-Start Evaluation")
        st.markdown("""
        Cold-start evaluation removes training edges touching held-out neurons,
        testing whether node-level information generalizes to previously unseen neurons.

        Graph heuristics collapse to ~0.50 because test nodes have zero training edges.
        """)

        cs = results.get("2b", {})
        if cs:
            rows = []
            model_map = {
                "rf_node_only": "RF Node Features",
                "rf_node_plus_heuristic": "RF + Heuristics",
                "gnn": "GraphSAGE",
                "random": "Random Baseline",
            }
            for key, display in model_map.items():
                if key in cs:
                    m = cs[key]
                    rows.append({
                        "Model": display,
                        "ROC-AUC": m.get("roc_auc"),
                        "PR-AUC": m.get("pr_auc"),
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
                st.bar_chart(df.set_index("Model")[["ROC-AUC", "PR-AUC"]])

            # Cold-start split sizes
            st.markdown("#### Cold-Start Split Sizes")
            split = cs.get("split_sizes", {})
            if split:
                split_df = pd.DataFrame([
                    {"Split": "Train", "Nodes": split.get("train_nodes"), "Edges": split.get("train_edges")},
                    {"Split": "Validation", "Nodes": split.get("val_nodes"), "Edges": split.get("val_edges")},
                    {"Split": "Test", "Nodes": split.get("test_nodes"), "Edges": split.get("test_edges")},
                ])
                st.dataframe(split_df.style.format({"Nodes": "{:,}", "Edges": "{:,}"}), hide_index=True)

    # --- Tab 3: Robustness (Exp 3) ---
    with tab3:
        st.markdown("## Experiment 3 — Robustness, Calibration, Plausibility")

        e3 = results.get("3", {})

        # Per-source ranking
        st.markdown("### Per-Source Ranking (3A)")
        psr = e3.get("3a_per_source_ranking", {})
        rbl = e3.get("3a_random_baseline", {})
        if psr and rbl:
            rows = [
                {"Metric": "Recall@10", "RF": psr.get("mean_recall@10"), "Random": rbl.get("mean_recall@10")},
                {"Metric": "Recall@50", "RF": psr.get("mean_recall@50"), "Random": rbl.get("mean_recall@50")},
                {"Metric": "Hit Rate@10", "RF": psr.get("hit_rate@10"), "Random": rbl.get("hit_rate@10")},
                {"Metric": "Hit Rate@50", "RF": psr.get("hit_rate@50"), "Random": rbl.get("hit_rate@50")},
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"RF": "{:.3f}", "Random": "{:.3f}"}), hide_index=True)

        # Score saturation
        st.markdown("### Score Saturation (3C)")
        sat = e3.get("3c_score_saturation", {})
        if sat:
            rows = []
            for threshold, info in sat.items():
                rows.append({
                    "Threshold": threshold,
                    "Count": info.get("count"),
                    "Percentage": f"{info.get('percentage', 0):.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        # Calibration
        st.markdown("### Calibration (3D)")
        cal = e3.get("3d_calibration", {})
        if cal:
            cols = st.columns(3)
            cols[0].metric("Brier Score", f"{cal.get('brier_score', 0):.4f}")
            cols[1].metric("Log Loss", f"{cal.get('log_loss', 0):.4f}")
            cols[2].metric("ECE", f"{cal.get('ece', 0):.4f}")

            st.markdown(
                "**Interpretation:** The RF model is more appropriate as a ranking "
                "system than as a calibrated probability estimator."
            )

            # Calibration curve
            cal_curve = cal.get("calibration_curve", [])
            if cal_curve:
                cal_df = pd.DataFrame(cal_curve)
                st.line_chart(cal_df.set_index("bin")[["mean_predicted", "observed_positive_rate"]])

        # Held-out rank distribution
        st.markdown("### Held-Out Positive Rank (3F)")
        rank_dist = e3.get("3f_rank_distribution", {})
        if rank_dist:
            cols = st.columns(3)
            cols[0].metric("Median Rank", f"{rank_dist.get('median_rank', 0):.1f}")
            cols[1].metric("Mean Rank", f"{rank_dist.get('mean_rank', 0):.1f}")
            cols[2].metric("Frac in Top-10", f"{rank_dist.get('frac_in_top_10', 0):.1%}")

            cols2 = st.columns(3)
            cols2[0].metric("Frac Ranked #1", f"{rank_dist.get('frac_ranked_1', 0):.1%}")
            cols2[1].metric("Frac in Top-50", f"{rank_dist.get('frac_in_top_50', 0):.1%}")

        # Exclusion audit
        st.markdown("### Exclusion Audit (3G)")
        excl = e3.get("3g_exclusion_audit", {})
        if excl:
            rows = [
                {"Check": "Self-loops", "Count": excl.get("self_loops", 0)},
                {"Check": "Known edges", "Count": excl.get("known_edges", 0)},
                {"Check": "Duplicate pairs", "Count": excl.get("duplicate_pairs", 0)},
                {"Check": "Invalid IDs", "Count": excl.get("invalid_ids", 0)},
                {"Check": "Non-finite scores", "Count": excl.get("non_finite_scores", 0)},
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    # --- Tab 4: Feature Importance ---
    with tab4:
        st.markdown("## Feature Importance")
        st.markdown(
            "Feature importance describes model behavior and should not be "
            "interpreted as proof of biological causation."
        )

        fi = store.get_feature_importance()
        if fi:
            fi_df = pd.DataFrame([
                {"Feature": f.feature, "Importance": f.importance, "Group": f.group}
                for f in fi
            ])
            st.dataframe(
                fi_df.style.format({"Importance": "{:.4f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(fi_df.set_index("Feature")["Importance"])

            # Group importance
            st.markdown("### Feature Group Importance")
            group_imp = fi_df.groupby("Group")["Importance"].sum().sort_values(ascending=False)
            st.bar_chart(group_imp)

        # Biological patterns
        st.markdown("### Biological Pattern Observations (3H)")
        bio = e3.get("3h_biological_patterns", {})
        if bio:
            st.markdown(
                "Neurotransmitter annotations are incomplete, so these patterns "
                "should be treated as exploratory observations rather than causal "
                "biological conclusions."
            )
            n_analyzed = bio.get("n_analyzed", 0)
            st.markdown(f"- **{n_analyzed}** top candidates analyzed")
            st.markdown("- **97.3%** have unknown NT types (annotation gap)")
            st.markdown("- Known-type pairs: GABA and ACH enriched")
