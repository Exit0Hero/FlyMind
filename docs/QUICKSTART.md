# FlyMind Quickstart

Get FlyMind running in 5 steps.

## 1. Requirements

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended)
- ~500 MB disk space for processed data

## 2. Dataset Setup

FlyMind requires the FlyWire FAFB connectome dataset.

1. Visit [flywire.ai](https://flywire.ai/)
2. Download the following files:
   - `neurons.csv.gz`
   - `connections_princeton.csv.gz`
   - `cell_stats.csv.gz`
   - `coordinates.csv.gz`
   - `classification.csv.gz`
   - `consolidated_cell_types.csv.gz`
   - `names.csv.gz`
   - `visual_neuron_types.csv.gz`
   - `connectivity_tags.csv.gz`
   - `processed_labels.csv.gz`
   - `column_assignment.csv.gz`
3. Place all files in a directory (e.g., `~/flywire-data/`)
4. Set the data path (see Step 3)

## 3. Environment Setup

```bash
# Clone the repository
git clone https://github.com/akkushon-kamen/flymind.git
cd flymind

# Install dependencies
pip install -r requirements.txt

# Set data path (optional, defaults to ../dataset/)
export FLYMIND_DATA_ROOT=~/flywire-data/
```

## 4. Processing

```bash
# Build neuron table
python src/build_neuron_table.py

# Build edge tables
python src/build_graph.py

# Run link prediction pipeline
python src/link_prediction/run_part1.py

# Run cold-start evaluation
python src/link_prediction/cold_start.py

# Run candidate ranking
python src/link_prediction/experiment_2d.py

# Run robustness analysis
python src/link_prediction/experiment_3.py
```

## 5. Testing

```bash
# Run fast tests (no raw data required)
python -m pytest tests/test_inference.py tests/test_dashboard.py -v

# Run full test suite (requires raw data)
python -m pytest tests/ -v
```

## 6. Dashboard Launch

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501

## Troubleshooting

**Data not found:**
- Verify `FLYMIND_DATA_ROOT` points to the correct directory
- Check that all required CSV files are present

**Memory errors:**
- Ensure at least 8 GB RAM available
- Close other memory-intensive applications

**Import errors:**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.10+ required)

**Dashboard won't start:**
- Verify Streamlit is installed: `pip install streamlit`
- Check port 8501 is not in use
