# FlyMind Data Card

## Dataset Source

FlyWire FAFB (Full Adult Fly Brain) connectome for *Drosophila melanogaster*.

**Citation:** Dorkenwald et al. (2024). FlyWire: A complete wiring diagram of the adult Drosophila brain. *Nature*.

**URL:** [https://flywire.ai/](https://flywire.ai/)

## Dataset Statistics

| Property | Value |
|----------|-------|
| Neurons | 139,255 |
| Directed edges | 3,732,460 |
| Brain regions (neuropils) | 76 |
| Neurotransmitter types | 6 (ACH, GABA, GLUT, DA, SER, OCT) |
| Super-classes | 10 |

## Primary Identifier

`root_id` — FlyWire-specific uint64 neuron identifier.

## Connection Direction

`pre_root_id → post_root_id` (directed, A→B ≠ B→A)

## Biological Annotations

### Neurotransmitter Types

| Type | Count | Percentage |
|------|------:|-----------:|
| ACH | 82,298 | 59.1% |
| GLUT | 19,605 | 14.1% |
| GABA | 16,017 | 11.5% |
| SER | 1,021 | 0.7% |
| DA | 584 | 0.4% |
| OCT | 72 | 0.1% |
| Unknown | 19,658 | 14.1% |

**Note:** NT labels are predictions, not experimental measurements.

### Morphological Features

- `length_nm`: Neurite length in nanometers
- `area_nm`: Soma surface area in nanometers
- `size_nm`: Soma size in nanometers

### Spatial Coordinates

- `coord_x`, `coord_y`, `coord_z`: Centroid body coordinates

### Classification Hierarchy

- `super_class`: High-level neuron class (10 classes)
- `class`: Intermediate classification
- `sub_class`: Fine-grained classification
- `primary_type`: Specific cell type label

## Super-Class Distribution

| Super-class | Count |
|-------------|------:|
| optic | 77,873 |
| central | 32,381 |
| sensory | 16,938 |
| visual_projection | 7,684 |
| ascending | 1,750 |
| descending | 1,305 |
| sensory_ascending | 612 |
| visual_centrifugal | 522 |
| motor | 110 |
| endocrine | 80 |

## Missing Values

- 27 neurons have missing morphology values (filled with 0)
- 19,658 neurons (14.1%) lack neurotransmitter type annotations

## Processing Pipeline

1. Raw CSV files → `build_neuron_table.py` → `neuron_table.parquet`
2. Raw CSV files → `build_graph.py` → `edges_aggregated.parquet`
3. Feature construction → `features.py` → `X_features.npy`

## Known Limitations

- Dataset-specific to FlyWire FAFB
- NT annotations are predictions, not measurements
- Morphology values may be incomplete
- Raw dataset is not redistributed in this repository

## License

License information should be verified from the original FlyWire dataset source.
