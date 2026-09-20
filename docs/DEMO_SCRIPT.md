# FlyMind Demo Script

**Duration:** 3–5 minutes

**Setup:** Run `streamlit run app/streamlit_app.py` before the demo. Enable "Demo Mode" in the sidebar.

---

## 0:00–0:30 — Problem Statement

> "FlyMind explores whether biological and morphological properties of individual neurons
> can predict directed connectivity in the fruit-fly connectome."

**Key points:**
- FlyWire FAFB dataset: 139,255 neurons, 3.73M directed edges
- Research question: Can neuron features predict connectivity, and does it generalize to unseen neurons?

**Show:** Overview page with key metrics (139,255 neurons, 3.73M edges, ~0.98 ROC-AUC)

---

## 0:30–1:00 — Approach

> "We represent each neuron using 15 biological and morphological features, then train
> a Random Forest to score source-target pairs."

**Key points:**
- 13 numeric features (morphology, spatial, neurotransmitter) + 2 categorical
- Pair engineering: source, target, difference, product → 60 dimensions
- Validated Random Forest (100 estimators)

**Show:** Architecture diagram on Overview page

---

## 1:00–2:00 — Neuron Explorer

> "Let's look at a specific neuron."

**Actions:**
1. Enable "Demo Mode" in sidebar → neuron `720575940597856265` loads automatically
2. Or type `720575940597856265` in the search box
3. Point out:
   - **Identity:** Tm16, ACH, optic super-class
   - **Morphology:** length, area, size
   - **Location:** centroid coordinates
   - **Connectivity:** outgoing/incoming connections

**Say:** "97.3% of neurons have unknown neurotransmitter annotations. This is a known limitation."

---

## 2:00–3:00 — Candidate Ranking

> "Now let's ask FlyMind to rank candidate target connections for this neuron."

**Actions:**
1. Go to "Candidate Ranking" page
2. Enter source: `720575940597856265`
3. Set K = 10
4. Show ranked table
5. Select top candidate
6. Show source vs target comparison

**Say:** "These are model-suggested candidate connections. A high ranking score indicates
statistical association in the evaluated model, not proof that a biological connection exists."

**Show:** The disclaimer about ranking scores not being literal probabilities.

---

## 3:00–4:00 — Validation

> "How do we know this works?"

**Actions:**
1. Go to "Research Results" → "Cold-Start" tab
2. Show cold-start ROC-AUC (~0.98) and PR-AUC (0.974)
3. Explain: "Training edges touching held-out neurons are excluded. The model is evaluated
   on previously unseen neurons."

**Say:** "Node-level biological and morphological information generalized strongly to
held-out neurons in this evaluation. This does not imply universal biological generalization."

---

## 4:00–5:00 — Research Takeaway

> "What did we learn?"

**Key points:**
1. Strong node-level predictive signal (ROC-AUC ~0.98)
2. Generalization to held-out neurons (cold-start)
3. Candidate ranking for hypothesis generation
4. Limitations: dataset-specific, not causal, requires biological validation

**Show:** About / Limitations page — read the "What FlyMind Does NOT Claim" section

**Closing:** "FlyMind provides a research interface for exploring connectivity hypotheses.
Biological validation requires wet-lab experiments."

---

## Verified Demo Neurons

| ID | Type | Super-class | NT | Use case |
|----|------|-------------|-----|----------|
| `720575940597856265` | Tm16 | optic | ACH | Standard demo neuron |
| `720575940602380768` | CB1538 | central | GABA | Central neuron example |
| `720575940609282825` | lLN2F_b | central | GABA | High connectivity |

## Tips

- Keep the "Demo Mode" checkbox enabled for quick access to example neurons
- If asked about GNN results, show the Cold-Start tab and explain RF outperformed GraphSAGE
- If asked about Experiment 2C, explain it was aborted due to memory constraints
- Never claim connections are "discovered" — always say "model-suggested candidates"
