# FlyWire Dataset Audit

**Generated:** 2026-09-19
**Dataset:** FlyWire FAFB (Full Adult Fly Brain) connectome
**Location:** `/home/akkushon-kamen/dataset/`

---

## 1. Dataset Inventory

| File | Format | Rows | Columns | Size (MB) |
|------|--------|------|---------|-----------|
| `synapse_coordinates.csv.gz` | csv.gz | 34,156,320 | 5 | 316.82 |
| `connections_buhmann_no_threshold.csv.gz` | csv.gz | 16,847,997 | 5 | 212.09 |
| `connections_princeton.csv.gz` | csv.gz | 5,342,446 | 5 | 68.46 |
| `coordinates.csv.gz` | csv.gz | 238,909 | 3 | 5.31 |
| `labels.csv.gz` | csv.gz | 160,045 | 9 | 4.77 |
| `neuropil_synapse_table.csv.gz` | csv.gz | 134,181 | 321 | 4.67 |
| `cell_stats.csv.gz` | csv.gz | 139,246 | 4 | 2.53 |
| `neurons.csv.gz` | csv.gz | 139,255 | 10 | 1.68 |
| `names.csv.gz` | csv.gz | 139,255 | 3 | 1.18 |
| `processed_labels.csv.gz` | csv.gz | 100,091 | 2 | 1.02 |
| `classification.csv.gz` | csv.gz | 139,255 | 8 | 0.93 |
| `consolidated_cell_types.csv.gz` | csv.gz | 138,327 | 3 | 0.90 |
| `connectivity_tags.csv.gz` | csv.gz | 134,437 | 2 | 0.64 |
| `visual_neuron_types.csv.gz` | csv.gz | 95,079 | 6 | 0.63 |
| `column_assignment.csv.gz` | csv.gz | 45,528 | 8 | 0.46 |
| `synapse_attachment_rates.csv.gz` | csv.gz | 162 | 5 | <0.01 |
| `GeneratedLabelledFlows.zip` | zip | N/A | N/A | 271.0 |

**Total raw data files:** 17 (16 CSV.gz + 1 ZIP)

---

## 2. Neuron Identifiers

**Primary identifier:** `root_id` (uint64 / Int64)

`root_id` is a FlyWire-specific neuron identifier that appears in **12 of 16** datasets:

| Dataset | Contains root_id |
|---------|:---:|
| cell_stats.csv.gz | Yes |
| classification.csv.gz | Yes |
| column_assignment.csv.gz | Yes |
| connectivity_tags.csv.gz | Yes |
| consolidated_cell_types.csv.gz | Yes |
| coordinates.csv.gz | Yes |
| labels.csv.gz | Yes |
| names.csv.gz | Yes |
| neurons.csv.gz | Yes |
| neuropil_synapse_table.csv.gz | Yes |
| processed_labels.csv.gz | Yes |
| visual_neuron_types.csv.gz | Yes |
| connections_princeton.csv.gz | No (uses `pre_root_id`/`post_root_id`) |
| connections_buhmann_no_threshold.csv.gz | No (uses `pre_root_id`/`post_root_id`) |
| synapse_coordinates.csv.gz | No (uses `pre_root_id`/`post_root_id`) |
| synapse_attachment_rates.csv.gz | No (neuropil-level summary) |

**FACT:** `root_id` is the correct neuron identifier. No `body_id` column exists in this dataset.

---

## 3. Connection Schema

### connections_princeton.csv.gz (Filtered — Recommended for Graph Construction)

| Column | Type | Description |
|--------|------|-------------|
| `pre_root_id` | Int64 | Pre-synaptic neuron root_id |
| `post_root_id` | Int64 | Post-synaptic neuron root_id |
| `neuropil` | string | Brain region abbreviation (e.g., ME_L, LO_R, GNG) |
| `syn_count` | Int64 | Number of synapses between pre and post neurons in this neuropil |
| `nt_type` | string | Neurotransmitter type (ACH, GABA, GLUT, DA, SER, OCT) |

**Direction:** Edges are directed (pre → post). No self-loops detected.
**Weight meaning:** `syn_count` represents the number of synapses in a specific neuropil for a given directed neuron pair.
**Duplicate check:** 0 duplicate rows.

### connections_buhmann_no_threshold.csv.gz (Unfiltered)

Same schema as Princeton but with 16.8M rows (3x more). This is the unfiltered version.

### synapse_coordinates.csv.gz

| Column | Type | Description |
|--------|------|-------------|
| `pre_root_id` | Int64 | Pre-synaptic neuron |
| `post_root_id` | Int64 | Post-synaptic neuron |
| `x` | Int64 | Synapse x-coordinate (nm) |
| `y` | Int64 | Synapse y-coordinate (nm) |
| `z` | Int64 | Synapse z-coordinate (nm) |

34.1M rows — very large. Contains spatial location of each synapse.

---

## 4. Dataset Sizes

### Neuron-level datasets

| Dataset | Total rows | Unique root_ids |
|---------|-----------|-----------------|
| neurons.csv.gz | 139,255 | 139,255 |
| names.csv.gz | 139,255 | 139,255 |
| classification.csv.gz | 139,255 | 139,255 |
| consolidated_cell_types.csv.gz | 138,327 | 138,327 |
| cell_stats.csv.gz | 139,246 | 139,246 |
| coordinates.csv.gz | 238,909 | 139,255 (duplicates exist — multiple supervoxels per neuron) |
| visual_neuron_types.csv.gz | 95,079 | 95,079 |
| connectivity_tags.csv.gz | 134,437 | ~134,437 (multi-tag neurons appear multiple times) |
| processed_labels.csv.gz | 100,091 | 100,091 |
| column_assignment.csv.gz | 45,528 | 45,528 |

### Edge-level datasets

| Dataset | Rows | Unique directed edges |
|---------|------|----------------------|
| connections_princeton.csv.gz | 5,342,446 | 3,732,460 |
| connections_buhmann_no_threshold.csv.gz | 16,847,997 | 15,091,983 |
| synapse_coordinates.csv.gz | 34,156,320 | N/A (one row per synapse) |

---

## 5. Join Coverage

Normalized neuron table built from `neurons.csv` as base, left-joining all annotation datasets:

| Column | Non-null | Coverage |
|--------|----------|----------|
| root_id | 139,255 | 100.0% |
| group | 139,255 | 100.0% |
| nt_type | 119,597 | 85.9% |
| nt_type_score | 139,255 | 100.0% |
| neurotransmitter scores (da/ser/gaba/glut/ach/oct_avg) | 139,255 | 100.0% |
| primary_type | 138,327 | 99.3% |
| additional_type(s) | 13,956 | 10.0% |
| flow | 139,255 | 100.0% |
| super_class | 139,255 | 100.0% |
| class | 107,591 | 77.3% |
| sub_class | 100,236 | 72.0% |
| hemilineage | 37,542 | 27.0% |
| side | 139,225 | 100.0% |
| nerve | 9,647 | 6.9% |
| length_nm | 139,246 | 100.0% |
| area_nm | 139,246 | 100.0% |
| size_nm | 139,246 | 100.0% |
| coord_x / coord_y / coord_z | 139,255 | 100.0% |
| name | 139,255 | 100.0% |
| visual type/family/subsystem/category | 95,079 | 68.3% |
| connectivity_tags | 134,437 | 96.5% |
| processed_labels | 100,091 | 71.9% |
| hemisphere / column_id / type (column assignment) | 45,528 | 32.7% |

---

## 6. Missing Data Summary

**High-coverage columns (>90%):** root_id, group, nt_type_score, neurotransmitter averages, flow, super_class, side, length/area/size, coordinates, name, primary_type, connectivity_tags

**Medium-coverage columns (50-90%):** nt_type (85.9%), class (77.3%), sub_class (72.0%), visual neuron types (68.3%), processed_labels (71.9%)

**Low-coverage columns (<50%):** hemilineage (27.0%), column_assignment fields (32.7%), nerve (6.9%), additional_type (10.0%)

---

## 7. Graph Statistics

Built from `connections_princeton.csv.gz` (filtered edges):

### Nodes
- **Total neurons in table:** 139,255
- **Neurons appearing as sources:** 137,518
- **Neurons appearing as targets:** 130,183
- **Total unique neurons in graph:** 138,584
- **Isolated neurons (in table but not in graph):** ~671

### Edges
- **Total edge rows:** 5,342,446
- **Unique directed edges:** 3,732,460
- **Self-loops:** 0
- **Duplicate edges:** 0
- **Directed:** Yes (pre → post)

### Edge Weights (synapse count)
| Statistic | Value |
|-----------|-------|
| Minimum | 1 |
| Maximum | 2,633 |
| Mean | 9.48 |
| Median | 6.00 |
| Std Dev | 14.78 |

### Degree Distribution
| Metric | Value |
|--------|-------|
| Out-degree min | 0 |
| Out-degree max | 14,000 |
| Out-degree mean | 38.55 |
| Out-degree median | 21.00 |
| In-degree min | 0 |
| In-degree max | 14,786 |
| In-degree mean | 38.55 |
| In-degree median | 16.00 |
| Total-degree mean | 77.10 |
| Total-degree max | 28,786 |

### Neurotransmitter Types in Edges
| NT Type | Edge Count |
|---------|-----------|
| ACH | 3,210,049 |
| GABA | 1,172,932 |
| GLUT | 826,380 |
| DA | 63,704 |
| SER | 40,396 |
| OCT | 28,985 |

### Top 10 Neuropils by Edge Count
| Neuropil | Edges |
|----------|-------|
| ME_R | 660,198 |
| ME_L | 616,856 |
| LO_R | 316,692 |
| LO_L | 306,759 |
| GNG | 226,326 |
| AVLP_R | 164,156 |
| LOP_R | 164,071 |
| AVLP_L | 128,730 |
| LOP_L | 115,527 |
| PVLP_L | 109,322 |

---

## 8. Potential ML Features

### Biological
- `primary_type` — cell type label (99.3% coverage)
- `super_class` — high-level class (100%)
- `class` — mid-level class (77.3%)
- `sub_class` — fine-grained class (72.0%)
- `flow` — intrinsic/afferent/efferent (100%)
- `hemilineage` — developmental lineage (27.0%)
- `nerve` — nerve assignment (6.9%)

### Neurotransmitter
- `nt_type` — dominant neurotransmitter (85.9%)
- `nt_type_score` — confidence score (100%)
- `da_avg`, `ser_avg`, `gaba_avg`, `glut_avg`, `ach_avg`, `oct_avg` — per-NT probability scores (100%)

### Morphological
- `length_nm` — neuron length in nanometers (100%)
- `area_nm` — surface area (100%)
- `size_nm` — volume proxy (100%)

### Spatial
- `coord_x`, `coord_y`, `coord_z` — neuron position (100%)
- `hemisphere`, `column_id`, `x`, `y` from column assignment (32.7%)

### Connectivity
- `connectivity_tags` — functional tags (96.5%)
- `connectivity_tag` — individual tags (multiple per neuron)
- `processed_labels` — community labels (71.9%)
- `name` — proofread neuron name (100%)
- `group` — neuron group (100%)

### Visual Neuron
- `type`, `family`, `subsystem`, `category`, `side` from visual_neuron_types (68.3%)

### Neuropil-level
- `neuropil_synapse_table.csv.gz` — 321 columns of per-neuropil input/output synapse counts and partner counts

---

## 9. Potential ML Tasks

### Task 1: Link Prediction

**Goal:** Predict whether neuron A connects to neuron B.

**Data required:** Edge table + node features.
**Feasibility:** HIGH. We have 3.7M directed edges with node features. Standard graph ML task.
**Challenges:** Graph is very dense (mean degree 77). Many possible edges. Requires negative sampling strategy.
**Tradeoffs:** Straightforward but may be less novel for a hackathon.

### Task 2: Neuron Type / Class Prediction

**Goal:** Predict `super_class`, `class`, or `primary_type` from graph structure + features.

**Data required:** Node features + graph structure.
**Feasibility:** HIGH. 100% coverage for super_class, 77% for class, 99.3% for primary_type. Rich features available.
**Challenges:** Class imbalance (some types very rare). Multi-label possible.
**Tradeoffs:** Clear evaluation metrics. Can use graph-aware models (GNN) or traditional ML on computed features.

### Task 3: Community / Functional Circuit Classification

**Goal:** Predict `connectivity_tags` or `processed_labels` from graph and biological features.

**Data required:** Node features + graph structure + community labels.
**Feasibility:** MEDIUM-HIGH. 96.5% coverage for connectivity_tags, 71.9% for processed_labels.
**Challenges:** Many possible tags. Need to define meaningful label grouping.
**Tradeoffs:** Highly interpretable — connects to real neuroscience.

### Task 4: Explainable Subgraph / Circuit Discovery

**Goal:** Find meaningful neural circuits (upstream/downstream pathways) using GNN explainability.

**Data required:** Full graph + node annotations.
**Feasibility:** MEDIUM. Requires GNN training + explainability methods (GNNExplainer, etc.).
**Challenges:** Evaluation is subjective. Hard to benchmark.
**Tradeoffs:** Most novel and impactful if it works, but hardest to validate.

### Task 5: Link Prediction with Neuropil Context

**Goal:** Predict neuropil-specific connections (which brain region a synapse occurs in).

**Data required:** Edge table with neuropil labels + node features.
**Feasibility:** MEDIUM-HIGH. Each edge has a neuropil label.
**Challenges:** Neuropil is per-edge, not per-neuron. Multi-task setup.
**Tradeoffs:** More specific than generic link prediction. Rich neuroscience interpretation.

---

## 10. Processed Files Generated

| File | Description |
|------|-------------|
| `data/processed/neuron_table.parquet` | Normalized neuron-level table (139,255 rows × 41 columns) |
| `data/processed/neuron_table.csv` | CSV version of neuron table |
| `data/processed/edges_princeton.parquet` | Edge table from Princeton connections (5,342,446 rows) |
| `data/processed/edges_buhmann.parquet` | Edge table from Buhmann connections (16,847,997 rows) |
| `results/figures/edge_weight_distribution.png` | Edge weight histogram |
| `results/figures/degree_distribution.png` | In/out-degree histograms |
| `results/figures/neuropil_distribution.png` | Top neuropils by edge count |
| `results/figures/spatial_scatter.png` | XY projection of neuron positions |
| `results/figures/annotation_coverage.png` | Feature coverage bar chart |
| `results/reports/file_audit.json` | Full machine-readable audit of all files |
| `results/reports/graph_stats.json` | Graph statistics JSON |

---

## 11. Test Results

All 9 data integrity tests **PASSED**:

| Test | Result |
|------|--------|
| neuron_table loads | PASS (139,255 rows, 41 cols) |
| root_id unique | PASS (139,255 unique) |
| root_id is integer | PASS (int64) |
| edge tables load | PASS (5.3M / 16.8M edges) |
| no self-loops | PASS |
| weights positive integers | PASS |
| source/target are integers | PASS |
| edge coverage > 90% | PASS (100.0% source, 100.0% target) |
| no data type corruption | PASS |

---

## 12. Recommended Next ML Experiments

Three technically feasible directions for the Fruit Fly hackathon, ordered by risk/reward:

### Direction 1: GNN-based Neuron Type Classification (Safest)

**Task:** Predict `super_class`, `class`, or `primary_type` from graph structure + node features.

| Aspect | Assessment |
|--------|-----------|
| Data readiness | HIGH — labels exist at 100%, 77%, 99.3% coverage |
| Model | GNN (GCN/GAT) with node features, or XGBoost on computed graph features |
| Evaluation | Accuracy, F1 (macro for imbalance), confusion matrix |
| Explainability | GNNExplainer to find which features/neighbors drive predictions |
| Hackathon fit | Strong — clear demo, clear metrics, well-understood task |
| Risk | Low |

**Tradeoffs:**
- (+) Cleanest evaluation. Labels are verified by neuroscientists.
- (+) Easy to ablate: graph features vs. biological features vs. both.
- (-) May not feel "novel" — node classification is a standard GNN benchmark.

### Direction 2: Neuropil-Aware Link Prediction (Most Unique)

**Task:** Predict which brain region (neuropil) a synapse occurs in, given two neurons.

| Aspect | Assessment |
|--------|-----------|
| Data readiness | HIGH — 5.3M edges with neuropil labels, 76 neuropils |
| Model | GNN with edge-level classification head, or factorization machine |
| Evaluation | Top-k accuracy, MAP, per-neuropil F1 |
| Explainability | Attention weights show which neuropils are predicted for a pair |
| Hackathon fit | Unique — nobody else is doing this. Strong 3D brain visualization. |
| Risk | Medium — 76-class edge classification is non-trivial |

**Tradeoffs:**
- (+) Most scientifically novel. Directly models brain-region specificity.
 (+) Beautiful visualization potential (3D fly brain with predicted neuropils).
- (-) Harder to evaluate. More complex model architecture.
- (-) Requires careful negative sampling (which neuropils are plausible?).

### Direction 3: Explainable Functional Circuit Discovery (Highest Impact)

**Task:** Train a GNN on `connectivity_tags`, then extract and explain discovered subcircuits.

| Aspect | Assessment |
|--------|-----------|
| Data readiness | MEDIUM-HIGH — 96.5% tag coverage, but many possible tags |
| Model | GAT/GIN with explainability layer (GNNExplainer, PGExplainer) |
| Evaluation | Qualitative (neuroscientist review) + quantitative (tag prediction) |
| Explainability | Core feature — extract influential subgraphs per tag |
| Hackathon fit | Maximum impact — "we discovered circuit X does Y" |
| Risk | High — hard to validate, subjective evaluation |

**Tradeoffs:**
- (+) Most impressive demo if it works.
- (+) Directly answers a neuroscience question.
- (-) Evaluation is inherently subjective.
- (-) Requires more careful feature engineering and explanation filtering.
- (-) May overfit to spurious patterns without careful regularization.

### Recommendation

**Start with Direction 1** (neuron classification) as the foundation — it builds the full pipeline (data loading, GNN training, evaluation) with low risk. Then **layer Direction 2 or 3 on top** to make it a standout hackathon project. The combination of "we can predict neuron types from connectivity" + "and here are the circuits we discovered" is the strongest narrative.
