"""Generate figures for Experiment 3: Robustness, Calibration, Plausibility."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")
FIG_DIR = RESULTS / "figures" / "experiment_3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "reports" / "experiment_3_robustness.json") as f:
    res = json.load(f)

# ── Figure 1: Score distribution + saturation ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Saturation bar chart
ax = axes[0]
sat = res["3c_score_saturation"]
thresholds = ["=1.0", ">=0.99", ">=0.95", ">=0.90", ">=0.75", ">=0.50"]
counts = [sat.get(f"score_{t}", {}).get("count", 0) for t in thresholds]
pcts = [sat.get(f"score_{t}", {}).get("percentage", 0) for t in thresholds]
colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
bars = ax.barh(thresholds[::-1], pcts[::-1], color=colors[::-1], height=0.6)
ax.set_xlabel("Percentage of Candidates (%)", fontsize=11)
ax.set_title("Score Saturation", fontsize=12, fontweight="bold")
for bar, pct in zip(bars, pcts[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{pct:.2f}%", va="center", fontsize=9)

# Calibration curve
ax = axes[1]
cal = res["3d_calibration"]["calibration_curve"]
mean_pred = [b["mean_predicted"] for b in cal if b["count"] > 0]
obs_rate = [b["observed_positive_rate"] for b in cal if b["count"] > 0]
n_samples = [b["count"] for b in cal if b["count"] > 0]
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")
ax.plot(mean_pred, obs_rate, "o-", color="#3498db", linewidth=2, markersize=6, label="RF Model")
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Observed Positive Rate", fontsize=11)
ax.set_title("Calibration Curve", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
# Annotate with sample sizes
for mp, or_, n in zip(mean_pred, obs_rate, n_samples):
    ax.annotate(f"n={n}", (mp, or_), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7, color="gray")

plt.tight_layout()
out = FIG_DIR / "experiment_3_calibration.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 2: Per-source ranking comparison (RF vs Random) ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

k_vals = [10, 50, 100]
rf_data = res["3a_per_source_ranking"]
rand_data = res["3a_random_baseline"]

# Recall@K
ax = axes[0]
rf_recall = [rf_data.get(f"mean_recall@{k}", 0) for k in k_vals]
rand_recall = [rand_data.get(f"mean_recall@{k}", 0) for k in k_vals]
x = np.arange(len(k_vals))
w = 0.35
ax.bar(x - w/2, rf_recall, w, label="RF Node Features", color="#2ecc71", edgecolor="white")
ax.bar(x + w/2, rand_recall, w, label="Random Baseline", color="#95a5a6", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in k_vals])
ax.set_ylabel("Mean Recall@K", fontsize=11)
ax.set_title("Recall@K: RF vs Random", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
for i, (r, rand) in enumerate(zip(rf_recall, rand_recall)):
    ax.text(i - w/2, r + 0.02, f"{r:.3f}", ha="center", fontsize=9, color="#27ae60")
    ax.text(i + w/2, rand + 0.02, f"{rand:.3f}", ha="center", fontsize=9, color="#7f8c8d")

# Hit Rate@K
ax = axes[1]
rf_hit = [rf_data.get(f"hit_rate@{k}", 0) for k in k_vals]
rand_hit = [rand_data.get(f"hit_rate@{k}", 0) for k in k_vals]
ax.bar(x - w/2, rf_hit, w, label="RF Node Features", color="#2ecc71", edgecolor="white")
ax.bar(x + w/2, rand_hit, w, label="Random Baseline", color="#95a5a6", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in k_vals])
ax.set_ylabel("Hit Rate@K", fontsize=11)
ax.set_title("Hit Rate@K: RF vs Random", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
for i, (h, rh) in enumerate(zip(rf_hit, rand_hit)):
    ax.text(i - w/2, h + 0.02, f"{h:.3f}", ha="center", fontsize=9, color="#27ae60")
    ax.text(i + w/2, rh + 0.02, f"{rh:.3f}", ha="center", fontsize=9, color="#7f8c8d")

plt.tight_layout()
out = FIG_DIR / "experiment_3_rank_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 3: Seed stability ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Score distributions across seeds
ax = axes[0]
seeds_data = res["3b_seed_stability"]
seed_ids = ["42", "43", "44", "45"]
means = [seeds_data.get(s, {}).get("score_stats", {}).get("mean", 0) for s in seed_ids]
medians = [seeds_data.get(s, {}).get("score_stats", {}).get("median", 0) for s in seed_ids]
p95s = [seeds_data.get(s, {}).get("score_stats", {}).get("p95", 0) for s in seed_ids]
x = np.arange(len(seed_ids))
w = 0.25
ax.bar(x - w, means, w, label="Mean", color="#3498db")
ax.bar(x, medians, w, label="Median", color="#2ecc71")
ax.bar(x + w, p95s, w, label="P95", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels([f"Seed {s}" for s in seed_ids])
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Score Distribution by Seed", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

# Jaccard similarity
ax = axes[1]
jaccard = seeds_data.get("jaccard", {})
pairs = list(jaccard.keys())
vals = list(jaccard.values())
if pairs:
    ax.barh(pairs[::-1], vals[::-1], color="#3498db", height=0.5)
    ax.set_xlabel("Jaccard Similarity", fontsize=11)
    ax.set_title("Top-100 Overlap Between Seeds", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.2 if vals else 1)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
else:
    ax.text(0.5, 0.5, "No Jaccard data", transform=ax.transAxes, ha="center")

plt.tight_layout()
out = FIG_DIR / "experiment_3_seed_stability.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 4: Feature importance ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Individual features
ax = axes[0]
feat_imp = res["feature_importance"]["top10"]
names = [f[0] for f in feat_imp][::-1]
imps = [f[1] for f in feat_imp][::-1]
ax.barh(names, imps, color="#3498db", edgecolor="white", height=0.6)
ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
ax.set_title("Top-10 Feature Importances", fontsize=12, fontweight="bold")

# Feature groups
ax = axes[1]
groups = res["feature_importance"]["groups"]
g_names = list(groups.keys())
g_vals = list(groups.values())
g_colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
ax.barh(g_names[::-1], g_vals[::-1], color=g_colors[::-1], height=0.5)
ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
ax.set_title("Feature Group Importances", fontsize=12, fontweight="bold")
for i, v in enumerate(g_vals[::-1]):
    ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
out = FIG_DIR / "experiment_3_biological_enrichment.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 5: Held-out positive rank distribution ──
fig, ax = plt.subplots(figsize=(8, 5))
rd = res["3f_rank_distribution"]
metrics = ["frac_ranked_1", "frac_in_top_10", "frac_in_top_50", "frac_in_top_100"]
labels = ["Ranked #1", "In Top-10", "In Top-50", "In Top-100"]
vals = [rd.get(m, 0) for m in metrics]
colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71"]
bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Fraction of Held-out Positives", fontsize=11)
ax.set_title("Held-out Positive Rank Distribution", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "experiment_3_score_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nAll figures generated.")
