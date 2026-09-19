# FlyMind

A machine-learning project using the FlyWire FAFB (Full Adult Fly Brain) *Drosophila melanogaster* connectome dataset.

## Dataset

This project uses connectome data from [FlyWire](https://flywire.ai/), a complete wiring diagram of the adult fruit-fly brain (*Drosophila melanogaster*), built from the FAFB (Full Adult Fly Brain) electron microscopy volume.

**Citation:** If you use this data, please cite the FlyWire consortium and the original FAFB dataset as described at [https://flywire.ai/](https://flywire.ai/).

### Raw Data Files

| File | Description |
|------|-------------|
| `neurons.csv.gz` | Neuron metadata with neurotransmitter type predictions |
| `classification.csv.gz` | Hierarchical neuron classification (super_class, class, sub_class) |
| `consolidated_cell_types.csv.gz` | Cell type labels |
| `cell_stats.csv.gz` | Morphological measurements (length, area, size) |
| `coordinates.csv.gz` | Neuron body coordinates |
| `names.csv.gz` | Proofread neuron names and groups |
| `visual_neuron_types.csv.gz` | Visual system neuron annotations |
| `connectivity_tags.csv.gz` | Functional connectivity tags |
| `processed_labels.csv.gz` | Community-refined labels |
| `column_assignment.csv.gz` | Columnar brain region assignments |
| `connections_princeton.csv.gz` | Filtered synaptic connections (5.3M edges) |
| `connections_buhmann_no_threshold.csv.gz` | Unfiltered synaptic connections (16.8M edges) |
| `synapse_coordinates.csv.gz` | Spatial locations of all synapses (34.1M rows) |
| `neuropil_synapse_table.csv.gz` | Per-neuron synapse counts by neuropil region |
| `synapse_attachment_rates.csv.gz` | Neuropil-level proofreading statistics |

### Processed Files

| File | Description |
|------|-------------|
| `data/processed/neuron_table.parquet` | Normalized neuron table (139,255 neurons × 41 features) |
| `data/processed/edges_princeton.parquet` | Directed edge table from Princeton connections |
| `data/processed/edges_buhmann.parquet` | Directed edge table from Buhmann connections |

### Key Statistics

- **Unique neurons:** 139,255
- **Filtered edges (Princeton):** 5,342,446 (3,732,460 unique directed)
- **Unfiltered edges (Buhmann):** 16,847,997 (15,091,983 unique directed)
- **Neuron identifier:** `root_id` (FlyWire-specific uint64)
- **Brain regions (neuropils):** 76
- **Neurotransmitter types:** ACH, GABA, GLUT, DA, SER, OCT

## Project Structure

```
flymind/
├── data/
│   ├── raw/          # Symlink or reference to original dataset
│   └── processed/    # Built neuron table and edge tables
├── src/
│   ├── inspect_data.py      # Comprehensive dataset audit
│   ├── build_neuron_table.py # Builds normalized neuron table
│   └── build_graph.py       # Builds edge tables + graph stats + plots
├── notebooks/
├── models/
├── results/
│   ├── figures/      # Generated plots
│   └── reports/      # Data audit report
├── tests/
│   └── test_data_integrity.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full dataset audit
python src/inspect_data.py

# Build normalized neuron table
python src/build_neuron_table.py

# Build edge tables and generate plots
python src/build_graph.py

# Run data integrity tests
python -m tests.test_data_integrity
```

## License

See metadata files in the downloaded dataset for licensing information.
