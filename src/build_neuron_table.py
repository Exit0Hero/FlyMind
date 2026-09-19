#!/usr/bin/env python3
"""Build a normalized neuron-level table by joining annotation files."""

import pathlib
import pandas as pd
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", **kwargs)


def build_neuron_table():
    print("Building neuron table...")

    # Start with neurons.csv as the base — it has root_id and NT info
    neurons = load_csv(DATA_DIR / "neurons.csv.gz")
    print(f"  neurons.csv: {len(neurons)} rows, {list(neurons.columns)}")

    # Consolidated cell types
    try:
        cell_types = load_csv(DATA_DIR / "consolidated_cell_types.csv.gz")
        print(f"  consolidated_cell_types.csv: {len(cell_types)} rows")
        # Check for duplicates
        if cell_types["root_id"].duplicated().any():
            print(f"    WARNING: {cell_types['root_id'].duplicated().sum()} duplicate root_ids")
            cell_types = cell_types.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(cell_types, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping consolidated_cell_types: {e}")

    # Classification
    try:
        classification = load_csv(DATA_DIR / "classification.csv.gz")
        print(f"  classification.csv: {len(classification)} rows")
        if classification["root_id"].duplicated().any():
            print(f"    WARNING: {classification['root_id'].duplicated().sum()} duplicate root_ids")
            classification = classification.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(classification, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping classification: {e}")

    # Cell stats
    try:
        cell_stats = load_csv(DATA_DIR / "cell_stats.csv.gz")
        print(f"  cell_stats.csv: {len(cell_stats)} rows")
        if cell_stats["root_id"].duplicated().any():
            cell_stats = cell_stats.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(cell_stats, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping cell_stats: {e}")

    # Coordinates
    try:
        coords_raw = load_csv(DATA_DIR / "coordinates.csv.gz")
        print(f"  coordinates.csv: {len(coords_raw)} rows")
            # coordinates has "position" which may be a string like "[x y z]" or "(x,y,z)"
        if "position" in coords_raw.columns:
            # Parse position strings - handles "[x y z]" and "(x,y,z)" formats
            import re
            def parse_position(pos):
                try:
                    if pd.isna(pos):
                        return None, None, None
                    s = str(pos).strip("[]()")
                    parts = re.split(r'[,\s]+', s)
                    parts = [float(x.strip()) for x in parts if x.strip()]
                    if len(parts) == 3:
                        return parts[0], parts[1], parts[2]
                except Exception:
                    pass
                return None, None, None

            coords_raw[["x", "y", "z"]] = coords_raw["position"].apply(
                lambda p: pd.Series(parse_position(p))
            )
            coords = coords_raw[["root_id", "x", "y", "z"]].copy()
            coords = coords.rename(columns={"x": "coord_x", "y": "coord_y", "z": "coord_z"})
        else:
            coords = coords_raw.copy()

        if "root_id" in coords.columns:
            coords = coords.drop_duplicates(subset="root_id", keep="first")
            neurons = neurons.merge(coords, on="root_id", how="left")
            print(f"    Parsed x,y,z from coordinates")
    except Exception as e:
        print(f"  Skipping coordinates: {e}")

    # Names
    try:
        names = load_csv(DATA_DIR / "names.csv.gz")
        print(f"  names.csv: {len(names)} rows")
        if names["root_id"].duplicated().any():
            names = names.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(names, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping names: {e}")

    # Visual neuron types
    try:
        visual = load_csv(DATA_DIR / "visual_neuron_types.csv.gz")
        print(f"  visual_neuron_types.csv: {len(visual)} rows")
        if visual["root_id"].duplicated().any():
            visual = visual.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(visual, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping visual_neuron_types: {e}")

    # Connectivity tags
    try:
        tags = load_csv(DATA_DIR / "connectivity_tags.csv.gz")
        print(f"  connectivity_tags.csv: {len(tags)} rows")
        # A neuron can have multiple tags; we'll join them
        tags_grouped = tags.groupby("root_id")["connectivity_tag"].apply(
            lambda x: "|".join(sorted(set(x.dropna())))
        ).reset_index()
        tags_grouped.columns = ["root_id", "connectivity_tags"]
        neurons = neurons.merge(tags_grouped, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping connectivity_tags: {e}")

    # Processed labels
    try:
        plabels = load_csv(DATA_DIR / "processed_labels.csv.gz")
        print(f"  processed_labels.csv: {len(plabels)} rows")
        if plabels["root_id"].duplicated().any():
            plabels = plabels.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(plabels, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping processed_labels: {e}")

    # Column assignment (hemisphere, type, column_id, x,y,p,q)
    try:
        col_assign = load_csv(DATA_DIR / "column_assignment.csv.gz")
        print(f"  column_assignment.csv: {len(col_assign)} rows")
        if col_assign["root_id"].duplicated().any():
            col_assign = col_assign.drop_duplicates(subset="root_id", keep="first")
        neurons = neurons.merge(col_assign, on="root_id", how="left")
    except Exception as e:
        print(f"  Skipping column_assignment: {e}")

    # Save
    out_parquet = OUT_DIR / "neuron_table.parquet"
    out_csv = OUT_DIR / "neuron_table.csv"

    neurons.to_parquet(out_parquet, index=False)
    print(f"\nSaved neuron table: {len(neurons)} rows, {len(neurons.columns)} columns")
    print(f"  Parquet: {out_parquet}")

    # CSV only if reasonably sized (< 100MB)
    if neurons.memory_usage(deep=True).sum() < 100e6:
        neurons.to_csv(out_csv, index=False)
        print(f"  CSV: {out_csv}")
    else:
        print(f"  CSV skipped (too large: {neurons.memory_usage(deep=True).sum()/1e6:.1f} MB)")

    # Coverage summary
    print("\n--- Coverage Summary ---")
    print(f"Total neurons: {len(neurons)}")
    for col in neurons.columns:
        non_null = neurons[col].notna().sum()
        pct = 100 * non_null / len(neurons)
        print(f"  {col:<35} {non_null:>8} ({pct:5.1f}%)")

    return neurons


if __name__ == "__main__":
    build_neuron_table()
