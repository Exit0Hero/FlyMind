# FlyMind Demo Script

**Duration:** 3–5 minutes
**Setup:** Run `streamlit run app/streamlit_app.py` before the demo.

---

## 1. Problem (30 seconds)

> "The fruit-fly connectome contains millions of directed connections between neurons. We wanted to know whether measurable properties of neurons can help predict which neurons are connected."

**Key points:**
- FlyWire FAFB dataset: 139,255 neurons
- 3.73M directed synaptic edges
- Research question: Can neuron features predict connectivity?

---

## 2. Dataset (30 seconds)

**Show:** Overview page with dataset statistics.

> "Each neuron is described by 15 biological and morphological features including neurotransmitter expression, neurite length, soma size, and spatial coordinates."

---

## 3. Research Result (45 seconds)

**Show:** Cold-start results on Overview page.

> "We trained a Random Forest model and evaluated it on previously unseen neurons—where training edges touching those neurons were excluded. The model achieved ROC-AUC 0.980 and PR-AUC 0.974."

Explain cold-start:
> "This means the model learned patterns from some neurons and was tested on completely different neurons it had never seen during training."

---

## 4. Live Neuron Exploration (45 seconds)

**Show:** Neuron Explorer page.

1. Enter neuron ID: `720575940597856265`
2. Show:
   - Neuron identity (Tm16, ACH, optic)
   - Morphology (length, area, size)
   - Connectivity summary

> "Each neuron has distinct biological properties. The model uses these properties to predict connectivity."

---

## 5. Connection Prediction (45 seconds)

**Show:** Connection Predictor page.

1. Source: `720575940597856265`
2. Target: `720575940602380768`
3. Show:
   - Connection status (observed/not observed)
   - Model ranking score
   - Source vs target comparison

> "The model assigns a ranking score to each pair. Higher scores indicate stronger statistical association in the model."

---

## 6. Candidate Ranking (45 seconds)

**Show:** Candidate Ranking page.

1. Source: `720575940597856265`
2. Show top 10 candidates
3. Select one candidate
4. Show source vs target comparison

> "These are model-suggested candidates for further investigation. They are not confirmed biological connections. Recall@10 is 0.814—meaning 81% of true connections appear in the top 10 candidates."

---

## 7. Limitations (30 seconds)

**Show:** About / Limitations page.

> "Important limitations: This is dataset-specific to FlyWire. 14.1% of neurons lack neurotransmitter annotations. Candidate connections require experimental validation. The model provides rankings, not biological proof."

---

## 8. Closing (15 seconds)

> "FlyMind does not claim to replace biological validation. It provides a scalable computational way to prioritize connectivity hypotheses for further investigation."

---

## Verified Demo Neurons

| ID | Type | Super-class | NT |
|----|------|-------------|-----|
| `720575940597856265` | Tm16 | optic | ACH |
| `720575940602380768` | CB1538 | central | GABA |
| `720575940609282825` | lLN2F_b | central | GABA |

## Tips

- Keep "Demo Mode" enabled in sidebar for quick access
- If asked about GNN: "GraphSAGE was evaluated comparatively but underperformed RF in these experiments"
- If asked about Experiment 2C: "It was aborted due to memory constraints"
- Never claim connections are "discovered" — always "model-suggested candidates"
