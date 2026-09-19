#!/usr/bin/env python3
"""Build edge table and compute graph statistics."""

import pathlib
import json
import pandas as pd
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
REPORT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path, **kw):
    return pd.read_csv(path, compression="gzip", **kw)


def build_edges():
    print("Building edge table from connections_princeton.csv...")
    edges = load_csv(DATA_DIR / "connections_princeton.csv.gz")
    print(f"  Raw rows: {len(edges)}")
    print(f"  Columns: {list(edges.columns)}")

    # Check schema
    print(f"\n  First 5 rows:")
    print(edges.head())

    # Check for duplicates
    dup_count = edges.duplicated().sum()
    print(f"\n  Duplicate rows: {dup_count}")

    # Check for self-loops
    self_loops = (edges["pre_root_id"] == edges["post_root_id"]).sum()
    print(f"  Self-loops: {self_loops}")

    # Unique neurons
    all_neuron_ids = set(edges["pre_root_id"].unique()) | set(edges["post_root_id"].unique())
    pre_only = set(edges["pre_root_id"].unique())
    post_only = set(edges["post_root_id"].unique())
    print(f"  Unique pre-synaptic neurons: {len(pre_only)}")
    print(f"  Unique post-synaptic neurons: {len(post_only)}")
    print(f"  Total unique neurons: {len(all_neuron_ids)}")

    # Directed edges
    directed_edges = set(zip(edges["pre_root_id"], edges["post_root_id"]))
    print(f"  Unique directed edges: {len(directed_edges)}")

    # Edge weight stats
    if "syn_count" in edges.columns:
        print(f"\n  Edge weight (syn_count) statistics:")
        print(f"    min:    {edges['syn_count'].min()}")
        print(f"    max:    {edges['syn_count'].max()}")
        print(f"    mean:   {edges['syn_count'].mean():.2f}")
        print(f"    median: {edges['syn_count'].median():.2f}")
        print(f"    std:    {edges['syn_count'].std():.2f}")

    # Degree distribution
    pre_degrees = edges.groupby("pre_root_id").size()
    post_degrees = edges.groupby("post_root_id").size()
    all_degrees = pd.concat([pre_degrees.rename("out"), post_degrees.rename("in")])
    total_degree = edges.groupby("pre_root_id").size().rename("out_deg")
    post_deg = edges.groupby("post_root_id").size().rename("in_deg")
    degree_df = pd.DataFrame({"out_deg": total_degree}).join(
        pd.DataFrame({"in_deg": post_deg}), how="outer"
    ).fillna(0).astype(int)
    degree_df["total"] = degree_df["out_deg"] + degree_df["in_deg"]

    print(f"\n  Degree statistics:")
    print(f"    out-degree min: {degree_df['out_deg'].min()}")
    print(f"    out-degree max: {degree_df['out_deg'].max()}")
    print(f"    out-degree mean: {degree_df['out_deg'].mean():.2f}")
    print(f"    out-degree median: {degree_df['out_deg'].median():.2f}")
    print(f"    in-degree min: {degree_df['in_deg'].min()}")
    print(f"    in-degree max: {degree_df['in_deg'].max()}")
    print(f"    in-degree mean: {degree_df['in_deg'].mean():.2f}")
    print(f"    in-degree median: {degree_df['in_deg'].median():.2f}")
    print(f"    total-degree min: {degree_df['total'].min()}")
    print(f"    total-degree max: {degree_df['total'].max()}")
    print(f"    total-degree mean: {degree_df['total'].mean():.2f}")

    # Neuropil distribution
    if "neuropil" in edges.columns:
        print(f"\n  Neuropil distribution:")
        neuropil_counts = edges["neuropil"].value_counts()
        for np_, cnt in neuropil_counts.items():
            print(f"    {np_:<10} {cnt}")

    # NT type distribution
    if "nt_type" in edges.columns:
        print(f"\n  NT type distribution:")
        nt_counts = edges["nt_type"].value_counts()
        for nt, cnt in nt_counts.items():
            print(f"    {nt:<10} {cnt}")

    # Save edge table
    edges_out = edges[["pre_root_id", "post_root_id", "syn_count"]].copy()
    edges_out.columns = ["source", "target", "weight"]
    edges_out.to_parquet(OUT_DIR / "edges_princeton.parquet", index=False)
    print(f"\nSaved edge table: {OUT_DIR / 'edges_princeton.parquet'}")

    # Also build buhmann edges
    print("\nBuilding edge table from connections_buhmann_no_threshold.csv...")
    buhmann = load_csv(DATA_DIR / "connections_buhmann_no_threshold.csv.gz")
    print(f"  Raw rows: {len(buhmann)}")
    print(f"  Unique edges: {len(buhmann.groupby(['pre_root_id', 'post_root_id']))}")

    buhmann_out = buhmann[["pre_root_id", "post_root_id", "syn_count"]].copy()
    buhmann_out.columns = ["source", "target", "weight"]
    buhmann_out.to_parquet(OUT_DIR / "edges_buhmann.parquet", index=False)
    print(f"Saved: {OUT_DIR / 'edges_buhmann.parquet'}")

    # Connectivity analysis
    stats = {
        "total_edges_princeton": len(edges),
        "total_edges_buhmann": len(buhmann),
        "unique_directed_edges_princeton": len(directed_edges),
        "unique_neurons_princeton": len(all_neuron_ids),
        "self_loops_princeton": int(self_loops),
        "syn_count_min": int(edges["syn_count"].min()),
        "syn_count_max": int(edges["syn_count"].max()),
        "syn_count_mean": float(edges["syn_count"].mean()),
        "syn_count_median": float(edges["syn_count"].median()),
        "out_degree_mean": float(degree_df["out_deg"].mean()),
        "out_degree_median": float(degree_df["out_deg"].median()),
        "in_degree_mean": float(degree_df["in_deg"].mean()),
        "in_degree_median": float(degree_df["in_deg"].median()),
    }

    with open(REPORT_DIR / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return edges, stats


def generate_plots():
    """Generate basic visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    edges = load_csv(DATA_DIR / "connections_princeton.csv.gz")

    # 1. Edge weight distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    weights = edges["syn_count"]
    ax.hist(weights, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Synapse Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Edge Weight Distribution (Princeton Connections)")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "edge_weight_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'edge_weight_distribution.png'}")

    # 2. Degree distribution
    pre_deg = edges.groupby("pre_root_id").size().values
    post_deg = edges.groupby("post_root_id").size().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(pre_deg, bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes[0].set_xlabel("Out-Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Out-Degree Distribution")
    axes[0].set_yscale("log")

    axes[1].hist(post_deg, bins=50, edgecolor="black", alpha=0.7, color="mediumseagreen")
    axes[1].set_xlabel("In-Degree")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("In-Degree Distribution")
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "degree_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'degree_distribution.png'}")

    # 3. Neuropil distribution
    if "neuropil" in edges.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        neuropil_counts = edges["neuropil"].value_counts().head(20)
        neuropil_counts.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
        ax.set_xlabel("Neuropil")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Top 20 Neuropils by Edge Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "neuropil_distribution.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {FIG_DIR / 'neuropil_distribution.png'}")

    # 4. Spatial scatter of neurons (if coordinates available)
    coords_file = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "neuron_table.parquet"
    if coords_file.exists():
        import pyarrow.parquet as pq
        neuron_table = pq.read_table(coords_file).to_pandas()
        if "x" in neuron_table.columns and neuron_table["x"].notna().sum() > 100:
            fig, ax = plt.subplots(figsize=(10, 8))
            valid = neuron_table.dropna(subset=["x", "y"])
            ax.scatter(valid["x"], valid["y"], s=0.5, alpha=0.3, c="navy")
            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Y (nm)")
            ax.set_title("Spatial Distribution of Neurons (XY projection)")
            ax.set_aspect("equal")
            plt.tight_layout()
            fig.savefig(FIG_DIR / "spatial_scatter.png", dpi=150)
            plt.close(fig)
            print(f"Saved: {FIG_DIR / 'spatial_scatter.png'}")

    # 5. Node coverage by annotation
    if coords_file.exists():
        import pyarrow.parquet as pq
        neuron_table = pq.read_table(coords_file).to_pandas()
        annotation_cols = [c for c in neuron_table.columns if c != "root_id"]
        coverage = {}
        for c in annotation_cols:
            coverage[c] = neuron_table[c].notna().sum()

        fig, ax = plt.subplots(figsize=(12, 6))
        pd.Series(coverage).sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="slateblue", edgecolor="black"
        )
        ax.set_ylabel("Non-Null Count")
        ax.set_title("Annotation Coverage Across Neuron Table Columns")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "annotation_coverage.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {FIG_DIR / 'annotation_coverage.png'}")


if __name__ == "__main__":
    edges, stats = build_edges()
    generate_plots()
    print("\nDone.")
