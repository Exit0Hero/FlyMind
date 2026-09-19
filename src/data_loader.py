"""Load neuron data and aggregate edges."""

import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR


def load_neuron_table() -> pd.DataFrame:
    """Load the normalized neuron table."""
    df = pd.read_parquet(PROCESSED_DIR / "neuron_table.parquet")
    return df


def load_and_aggregate_edges() -> pd.DataFrame:
    """Load edges and aggregate by (source, target), summing weights."""
    edges = pd.read_parquet(PROCESSED_DIR / "edges_princeton.parquet")
    aggregated = edges.groupby(["source", "target"], sort=False)["weight"].sum().reset_index()
    return aggregated


def get_edge_stats(edges: pd.DataFrame) -> dict:
    """Compute basic edge statistics."""
    return {
        "total_rows": len(edges),
        "unique_sources": edges["source"].nunique(),
        "unique_targets": edges["target"].nunique(),
        "total_neurons": len(set(edges["source"]) | set(edges["target"])),
        "weight_min": int(edges["weight"].min()),
        "weight_max": int(edges["weight"].max()),
        "weight_mean": float(edges["weight"].mean()),
        "weight_median": float(edges["weight"].median()),
    }
