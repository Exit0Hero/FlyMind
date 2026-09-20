"""Candidate Ranking page — rank candidate target connections for a source neuron."""

import streamlit as st
import pandas as pd
from app.services import get_store


def render(demo_id=None):
    store = get_store()
    df = store.neuron_table

    st.markdown("# Candidate Ranking")
    st.markdown("Rank candidate target neurons for a given source neuron based on model scoring.")

    # --- Source Selection ---
    st.markdown("### Select Source Neuron")
    source_id = st.text_input(
        "Source neuron root_id",
        value=str(demo_id) if demo_id else "",
        placeholder="e.g. 720575940597856265",
    )

    if not source_id:
        st.info("Enter a source neuron ID to rank candidate targets.")
        return

    try:
        src = int(source_id)
    except ValueError:
        st.error("Invalid root_id. Must be an integer.")
        return

    src_data = store.get_neuron(src)
    if src_data is None:
        st.error(f"Neuron {src} not found.")
        return

    # Show source info
    st.markdown(f"**Source:** `{src}` — {src_data.primary_type or '—'} ({src_data.super_class or '—'})")

    # --- Ranking Parameters ---
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of candidates (K)", 5, 100, 20)
    with col2:
        exclude_observed = st.checkbox("Exclude observed connections", value=True)

    # --- Rank Candidates ---
    with st.spinner("Ranking candidates..."):
        results = store.rank_candidate_targets(src, k=k, exclude_observed=exclude_observed)

    if not results:
        st.warning("No candidates found.")
        return

    # Convert dataclass objects to dicts for DataFrame
    results_dicts = [{
        "rank": r.rank,
        "source": str(r.source_root_id),
        "target": str(r.target_root_id),
        "ranking_score": r.score,
        "target_primary_type": r.target_primary_type or "—",
        "target_super_class": r.target_super_class or "—",
        "target_nt_type": r.target_nt_type or "—",
    } for r in results]
    results_df = pd.DataFrame(results_dicts)

    st.markdown("---")
    st.markdown(f"## Top {len(results_df)} Candidate Connections")

    # Summary metrics
    if len(results_df) > 0:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Top Score", f"{results_df['ranking_score'].iloc[0]:.4f}")
        mc2.metric("Median Score", f"{results_df['ranking_score'].median():.4f}")
        mc3.metric("Score Range", f"{results_df['ranking_score'].min():.4f} – {results_df['ranking_score'].max():.4f}")

    st.markdown("""
    <div class="disclaimer">
    <strong>Interpretation:</strong> These are model-suggested candidate connections, not confirmed
    biological discoveries. Ranking scores are used for ordering candidates and should not be
    interpreted as literal biological probabilities.
    </div>
    """, unsafe_allow_html=True)

    # Candidate table
    display_df = results_df[["rank", "target", "target_primary_type", "target_super_class", "ranking_score"]].copy()
    display_df.columns = ["Rank", "Target ID", "Type", "Super-class", "Score"]
    st.dataframe(display_df, use_container_width=True, height=min(400, 40 + len(display_df) * 35))

    # --- Score Distribution ---
    st.markdown("## Score Distribution")
    st.bar_chart(results_df.set_index("rank")["ranking_score"])

    # --- Candidate Detail ---
    st.markdown("---")
    st.markdown("## Candidate Detail")
    candidate_options = [f"{r.rank}. {r.target_root_id} — {r.target_primary_type or '?'} (score: {r.score:.4f})" for r in results]
    selected = st.selectbox("Select a candidate to inspect", candidate_options)

    if selected:
        idx = int(selected.split(".")[0]) - 1
        cand = results[idx]
        tgt_id = cand.target_root_id
        tgt_data = store.get_neuron(tgt_id)

        st.markdown("---")
        st.markdown("## Source vs Target Comparison")

        scol, tcol = st.columns(2)

        def _card(col, data, label):
            with col:
                st.markdown(f"**{label}**")
                name_val = data.name if pd.notna(data.name) else "—"
                st.markdown(f"**Root ID:** `{data.root_id}`")
                st.markdown(f"**Name:** {name_val}")
                st.markdown(f"**Super-class:** {data.super_class or '—'}")
                st.markdown(f"**Primary Type:** {data.primary_type or '—'}")
                nt = data.nt_type
                st.markdown(f"**Neurotransmitter:** `{nt if pd.notna(nt) else 'Unknown'}`")
                for key, label_m in [("length_nm", "Length"), ("area_nm", "Area"), ("size_nm", "Size")]:
                    val = getattr(data, key, None)
                    if pd.notna(val) and val > 0:
                        st.markdown(f"**{label_m}:** {val:,.0f} nm")
                    else:
                        st.markdown(f"**{label_m}:** —")
                for key, label_c in [("coord_x", "X"), ("coord_y", "Y"), ("coord_z", "Z")]:
                    val = getattr(data, key, None)
                    if pd.notna(val):
                        st.markdown(f"**{label_c}:** {val:,.1f}")

        _card(scol, src_data, "SOURCE NEURON")
        if tgt_data is not None:
            _card(tcol, tgt_data, "TARGET NEURON")
        else:
            with tcol:
                st.error("Target neuron data not found.")

        # Connection status
        is_obs = store.check_connection(src, tgt_id)
        if is_obs:
            st.success("**Observed in evaluated dataset**")
        else:
            st.info("**Not observed in evaluated dataset**")

        st.metric("Ranking Score", f"{cand.score:.4f}")

        # Pair features
        src_arr = store.get_feature_vector(src)
        tgt_arr = store.get_feature_vector(tgt_id)
        if src_arr is not None and tgt_arr is not None:
            feature_names = ["length_nm", "area_nm", "size_nm", "coord_x", "coord_y", "coord_z",
                             "nt_type_score", "ach_avg", "gaba_avg", "glut_avg", "da_avg", "ser_avg", "oct_avg",
                             "synapse_count", "total_nt_score"]
            pair_data = {}
            for i, fname in enumerate(feature_names):
                if i < len(src_arr) and i < len(tgt_arr):
                    pair_data[fname] = {
                        "Source": f"{src_arr[i]:.4f}",
                        "Target": f"{tgt_arr[i]:.4f}",
                        "|Diff|": f"{abs(src_arr[i] - tgt_arr[i]):.4f}",
                        "Product": f"{src_arr[i] * tgt_arr[i]:.4f}",
                    }
            st.markdown("### Pair-Level Features")
            st.dataframe(pd.DataFrame(pair_data).T, use_container_width=True)

        # Quick actions
        st.markdown("---")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Explore this target neuron", use_container_width=True):
                st.session_state["explorer_id"] = str(tgt_id)
                st.switch_page("app/streamlit_app.py")
        with a2:
            if st.button("Predict connection for this pair", use_container_width=True):
                st.session_state["predictor_source"] = source_id
                st.session_state["predictor_target"] = str(tgt_id)
                st.switch_page("app/streamlit_app.py")
