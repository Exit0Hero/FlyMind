"""About / Methodology page — scientific context and known limitations."""

import streamlit as st
from app.theme import inject_css, render_section_label, render_badge, render_disclaimer

def render():
    inject_css()

    st.markdown("# About / Methodology")
    st.markdown('<p style="color:#A6ADBB; font-size:14px">Scientific context, methodology, and known limitations of FlyMind.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Research question
    st.markdown(render_section_label("RESEARCH QUESTION"), unsafe_allow_html=True)
    st.markdown("""
    <div class="flymind-card">
        <p style="font-size:16px; color:#EDEFF4; line-height:1.6; margin:0">
            Can neuron-level biological and morphological properties predict directed
            connectivity in the fruit-fly connectome, and do those relationships
            generalize to previously unseen neurons?
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # What FlyMind does NOT claim
    st.markdown(render_section_label("WHAT FLYMIND DOES NOT CLAIM"), unsafe_allow_html=True)
    st.markdown("""
    <div class="flymind-card" style="border-color:rgba(229,83,75,0.3)">
        <ul style="color:#A6ADBB; line-height:2; margin:0; padding-left:20px">
            <li>Prove an unobserved connection exists</li>
            <li>Discover biological truth automatically</li>
            <li>Predict fly behavior</li>
            <li>Establish neurotransmitter causality</li>
            <li>Reconstruct the entire connectome</li>
            <li>Provide literal biological probabilities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # What FlyMind DOES
    st.markdown(render_section_label("WHAT FLYMIND DOES"), unsafe_allow_html=True)
    st.markdown("""
    <div class="flymind-card" style="border-color:rgba(79,209,197,0.3)">
        <ul style="color:#A6ADBB; line-height:2; margin:0; padding-left:20px">
            <li>Learn statistical relationships from the evaluated connectome</li>
            <li>Test generalization to held-out neurons</li>
            <li>Rank candidate connection hypotheses</li>
            <li>Provide an interactive research interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # How It Works
    st.markdown(render_section_label("HOW IT WORKS"), unsafe_allow_html=True)
    steps = [
        ("1", "Represent", "each neuron using biological and morphological features (15 base features)"),
        ("2", "Construct", "source-target feature pairs (pair, difference, interaction — 60 dimensions)"),
        ("3", "Train", "a validated Random Forest on observed connectivity"),
        ("4", "Evaluate", "on held-out neurons (cold-start evaluation)"),
        ("5", "Rank", "candidate target neurons for any source neuron"),
        ("6", "Present", "high-ranking pairs as hypotheses for further investigation"),
    ]
    for num, verb, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:12px">
            <div style="min-width:32px; height:32px; border-radius:50%; background:rgba(79,209,197,0.15); display:flex; align-items:center; justify-content:center; font-family:'Inter Tight',sans-serif; font-weight:700; font-size:14px; color:#4FD1C5">{num}</div>
            <div>
                <span style="font-weight:600; color:#EDEFF4">{verb}</span>
                <span style="color:#A6ADBB"> {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Limitations
    st.markdown(render_section_label("KNOWN LIMITATIONS"), unsafe_allow_html=True)

    with st.expander("Dataset", expanded=True):
        st.markdown("""
        - 139,255 neurons from FlyWire FAFB connectome
        - 3,732,460 unique directed edges
        - Dataset-specific coverage and filtering
        - **14.1% of neurons (19,658) lack neurotransmitter annotations**
        - Annotation quality varies across brain regions
        """)

    with st.expander("Model", expanded=True):
        st.markdown("""
        - Random Forest is the validated primary model (100 estimators, 60 features)
        - GraphSAGE was evaluated comparatively and performed worse
        - Model captures statistical correlations, not causal mechanisms
        - 15 base features + pair engineering = 60 dimensions
        """)

    with st.expander("Generalization", expanded=True):
        st.markdown("""
        - Cold-start evaluation provides evidence for generalization within the evaluated setup
        - Training edges touching held-out neurons are excluded
        - **Does not imply universal biological generalization**
        - Generalization is specific to this dataset, feature set, and evaluation protocol
        """)

    with st.expander("Biology", expanded=True):
        st.markdown("""
        - Model associations are not causal biological mechanisms
        - Neurotransmitter annotations are predicted, not measured for most neurons
        - Spatial proximity may confound connectivity predictions
        - Biological validation requires wet-lab experiments
        """)

    with st.expander("Candidate Interpretation", expanded=True):
        st.markdown("""
        - Candidate rankings are hypotheses for investigation
        - A high ranking score does not prove a biological connection exists
        - Ranking scores should not be interpreted as literal biological probabilities
        - Biological validation requires experimental confirmation
        """)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Terminology
    st.markdown(render_section_label("TERMINOLOGY GUIDE"), unsafe_allow_html=True)
    terms = [
        ("Root ID", "Unique FlyWire neuron identifier"),
        ("Super-class", "High-level biological classification (e.g., motor, sensory, central)"),
        ("Primary Type", "Specific neuron type label"),
        ("Ranking Score", "Model-derived score for candidate ordering"),
        ("Cold-start", "Evaluation on previously unseen neurons"),
        ("Observed connection", "Directed edge present in the evaluated graph"),
        ("Candidate connection", "Model-suggested pair not present in the evaluated graph"),
        ("Pair features", "Features constructed from source + target neuron properties"),
        ("ECE", "Expected Calibration Error — measures probability calibration"),
        ("Recall@K", "Fraction of true connections in top-K ranked candidates"),
    ]
    for term, definition in terms:
        st.markdown(f"""
        <div style="display:flex; gap:16px; padding:8px 0; border-bottom:1px solid #232A38">
            <span style="min-width:160px; font-weight:600; color:#EDEFF4; font-size:13px">{term}</span>
            <span style="color:#A6ADBB; font-size:13px">{definition}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Citation
    st.markdown(render_section_label("DATASET CITATION"), unsafe_allow_html=True)
    st.markdown("""
    <div class="flymind-card">
        <p style="color:#A6ADBB; font-size:14px; line-height:1.6; margin:0">
            This project uses connectome data from
            <a href="https://flywire.ai/" target="_blank" style="color:#4FD1C5">FlyWire</a>,
            a complete wiring diagram of the adult fruit-fly brain (Drosophila melanogaster),
            built from the FAFB (Full Adult Fly Brain) electron microscopy volume.
        </p>
    </div>
    """, unsafe_allow_html=True)
