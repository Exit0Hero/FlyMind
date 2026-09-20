# FlyMind — Judge Q&A

Prepared answers for likely questions during evaluation.

---

### Why Random Forest?

Random Forest was chosen as the primary model because:
- Handles mixed feature types (numeric + categorical) well
- Provides interpretable feature importance
- Strong baseline for tabular data
- No graph structure required for cold-start evaluation
- validated Random Forest outperformed GraphSAGE in our experiments

---

### Why not GNN?

GraphSAGE was evaluated as a comparative model. In our cold-start experiments:
- Random Forest ROC-AUC: ~0.98
- GraphSAGE ROC-AUC: 0.6485

The Random Forest model was stronger in the evaluated link-prediction experiments.
This does not mean GNNs are inherently inferior — it is specific to our data, features,
and evaluation setup.

---

### Why cold-start evaluation?

Cold-start evaluation tests whether the model generalizes to **previously unseen neurons**.
This is the scientifically meaningful question: can the model learn transferable patterns
about neuron connectivity, or does it merely memorize specific neurons?

---

### How did you avoid leakage?

- Training edges touching held-out neurons are **excluded**
- Evaluation is performed entirely on held-out neuron pairs
- Negative sampling uses compact hash-based edge index to verify non-existence
- No information from test neurons leaks into training

---

### How were negative examples generated?

Negative examples are randomly sampled pairs that are **not present** in the evaluated
connectivity graph. The compact hash-based edge index (numpy memmap) enables efficient
lookup without loading the full edge list into memory.

---

### What does the ranking score mean?

The ranking score is a model-derived score used for ordering candidate connections.
A higher score indicates the model ranks this pair as more likely to have a directed
connection based on learned statistical patterns.

**Important:** The score should NOT be interpreted as a literal biological probability.

---

### Why isn't a high score proof of a connection?

The model learns statistical associations from the evaluated connectome.
- Spatial proximity may confound predictions
- The model does not establish causal mechanisms
- Biological validation requires wet-lab experiments
- The score reflects model confidence, not biological certainty

---

### What happens with unknown neurotransmitter annotations?

97.3% of neurons have unknown neurotransmitter annotations. The model uses:
- `nt_type_score` (confidence in prediction)
- Individual NT sub-scores (ach_avg, gaba_avg, etc.)

Unknown annotations are handled gracefully — the model does not require complete
annotation to make predictions.

---

### Why was Experiment 2C not completed?

Experiment 2C was aborted due to **memory constraints** (RAM exhaustion).
It is not treated as a scientific result. No partial or incomplete result from
Experiment 2C is presented in the dashboard or research artifacts.

---

### What makes the candidate ranking useful?

Candidate ranking provides a **hypothesis generation tool**:
- Prioritizes which connections to investigate
- Uses biological and morphological features
- Generalizes to held-out neurons
- Enables interactive exploration

It does **not** prove connections exist — it generates testable hypotheses.

---

### What would biological validation require?

Biological validation would require:
- Electron microscopy confirmation of synaptic connections
- Physiological recording of functional connectivity
- Genetic manipulation experiments
- Cross-validation with independent datasets

FlyMind generates hypotheses; biology confirms them.

---

### Can this predict fly behavior?

**No.** FlyMind predicts directed connectivity between neurons based on biological
and morphological features. It does not model neural circuits, information processing,
or behavior.

---

### Can this reconstruct the complete connectome?

**No.** FlyMind:
- Uses 3.73M edges out of a much larger potential edge space
- Is trained on dataset-specific connectivity patterns
- Does not claim to reconstruct the complete connectome
- Provides candidate hypotheses, not complete connectivity maps

---

### How does the pair feature engineering work?

For each source-target pair, we construct:
- **Source features:** 15 base features of source neuron
- **Target features:** 15 base features of target neuron
- **Difference:** |source - target| for each feature
- **Product:** source × target for each feature

This gives 60 dimensions total, capturing individual neuron properties and their relationship.

---

### What features matter most?

Top features by importance:
1. `area_nm` — neuron surface area
2. `length_nm` — neurite length
3. `size_nm` — soma size
4. `coord_x`, `coord_y` — spatial position
5. Neurotransmitter scores (gaba_avg, ach_avg, da_avg)

Feature importance describes model behavior, not biological causation.
