"""Local neighborhood analysis for model interpretation."""

import numpy as np
import torch
import pandas as pd
from collections import Counter


def analyze_neuron(
    model, data, neuron_idx, node_ids, df,
    id_to_idx, target_classes, device,
    top_k_neighbors=10,
):
    """Analyze a single neuron's local neighborhood.

    Returns a human-readable report.
    """
    model.eval()

    # Get prediction
    with torch.no_grad():
        x = data.x.unsqueeze(0).to(device) if data.x.dim() == 1 else data.x.to(device)
        z = model.sage(x, data.edge_index.to(device))
        logits = model.classifier(z)

    logits_np = logits.cpu().numpy()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    pred_class_idx = int(np.argmax(logits_np[neuron_idx]))
    true_class_idx = int(data.y[neuron_idx].item())
    confidence = float(probs[neuron_idx, pred_class_idx])

    root_id = int(node_ids[neuron_idx])
    true_class = target_classes[true_class_idx]
    pred_class = target_classes[pred_class_idx]

    # Find neighbors
    edge_index = data.edge_index.cpu().numpy()
    edge_weights = data.edge_attr.cpu().numpy().flatten() if data.edge_attr is not None else None

    # Incoming edges (neighbors → this neuron)
    incoming = edge_index[1] == neuron_idx
    incoming_src = edge_index[0, incoming]
    incoming_weights = edge_weights[incoming] if edge_weights is not None else np.ones(incoming.sum())

    # Outgoing edges (this neuron → neighbors)
    outgoing = edge_index[0] == neuron_idx
    outgoing_tgt = edge_index[1, outgoing]
    outgoing_weights = edge_weights[outgoing] if edge_weights is not None else np.ones(outgoing.sum())

    # Sort by weight
    in_order = np.argsort(-incoming_weights)[:top_k_neighbors]
    out_order = np.argsort(-outgoing_weights)[:top_k_neighbors]

    # Build report
    lines = []
    lines.append(f"Target neuron index: {neuron_idx}")
    lines.append(f"Root ID: {root_id}")
    lines.append(f"True class: {true_class}")
    lines.append(f"Predicted class: {pred_class}")
    lines.append(f"Confidence: {confidence:.4f}")
    lines.append(f"Correct: {'Yes' if true_class == pred_class else 'No'}")
    lines.append("")
    lines.append(f"--- Top {top_k_neighbors} incoming neighbors ---")
    for i in in_order:
        src_idx = incoming_src[i]
        w = incoming_weights[i]
        src_root = int(node_ids[src_idx])
        src_class = target_classes[int(data.y[src_idx].item())]
        lines.append(f"  Neuron {src_root} ({src_class}) → Target | synapses: {w:.0f}")

    lines.append("")
    lines.append(f"--- Top {top_k_neighbors} outgoing neighbors ---")
    for i in out_order:
        tgt_idx = outgoing_tgt[i]
        w = outgoing_weights[i]
        tgt_root = int(node_ids[tgt_idx])
        tgt_class = target_classes[int(data.y[tgt_idx].item())]
        lines.append(f"  Target → Neuron {tgt_root} ({tgt_class}) | synapses: {w:.0f}")

    return "\n".join(lines)
