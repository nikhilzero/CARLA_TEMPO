#!/usr/bin/env python3
"""
plot_results.py — Generate thesis-quality visualizations of CARLA closed-loop eval results.

Data: Town05 routes S16-S25 (10 routes), 5 batches × 2 routes each.
Run: python scripts/plot_results.py
Output: figures/ directory (PDF + PNG for each plot)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
ROUTE_LABELS = [f"S{i}" for i in range(16, 26)]

data = {
    "Baseline":       [0.406,  4.910,  2.788,  6.192,  5.960, 11.864,  1.806,  0.397,  6.291,  1.898],
    "T=2, d=0.3":    [3.399, 12.014,  4.391,  4.472,  0.222, 16.393,  3.402,  8.415,  0.764,  3.246],
    "T=4, d=0.3":    [9.215,  7.578,  3.035, 11.291,  0.034, 24.393,  5.997, 10.393,  5.801,  3.833],
    "Stride=1, d=0.3": [1.973, 8.410,  4.253,  3.348,  0.061, 22.003,  3.934,  5.004,   None,   None],
    # d=0.1 models — all DS=0.0
    "T=2, d=0.1":    [0.0]*10,
    "T=4, d=0.1":    [0.0]*10,
    "T=8, d=0.1":    [0.0]*10,
    "Stride=1, d=0.1": [0.0]*10,
    "CrossAttn, d=0.1": [0.0]*10,
}

def avg(vals):
    v = [x for x in vals if x is not None]
    return np.mean(v) if v else 0.0

AVGS = {k: avg(v) for k, v in data.items()}

# Colors
BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#1a9850"
ORANGE = "#f46d43"
PURPLE = "#762a83"
GRAY   = "#969696"

# ── Plot 1: Average DS bar chart (all models) ─────────────────────────────────
def plot_avg_ds():
    models = [
        "Baseline",
        "T=2, d=0.1", "T=2, d=0.3",
        "T=4, d=0.1", "T=4, d=0.3",
        "T=8, d=0.1",
        "Stride=1, d=0.1", "Stride=1, d=0.3",
        "CrossAttn, d=0.1",
    ]
    avgs = [AVGS[m] for m in models]
    colors = [GRAY, RED, GREEN, RED, GREEN, RED, RED, GREEN, RED]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(models)), avgs, color=colors, edgecolor="white", linewidth=0.8, width=0.65)

    # Value labels
    for bar, val in zip(bars, avgs):
        if val > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.35,
                    "0.00", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="white")

    # Baseline reference line
    ax.axhline(AVGS["Baseline"], color=GRAY, linestyle="--", linewidth=1.2, alpha=0.7, label=f"Baseline ({AVGS['Baseline']:.2f})")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Avg Driving Score (DS)", fontsize=12)
    ax.set_title("CARLA Closed-Loop Evaluation — Average Driving Score by Model\n"
                 "(Town05, 10 routes; ✦ = d=0.3 models drive; d=0.1 models DS=0.0)", fontsize=11)
    ax.set_ylim(0, max(avgs) * 1.18)

    legend_patches = [
        mpatches.Patch(color=GREEN, label="dropout=0.3 (drives)"),
        mpatches.Patch(color=RED,   label="dropout=0.1 (never moves)"),
        mpatches.Patch(color=GRAY,  label="Baseline"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9.5)

    plt.tight_layout()
    _save(fig, "fig1_avg_ds_all_models")

# ── Plot 2: Per-route heatmap ─────────────────────────────────────────────────
def plot_heatmap():
    hmap_models = ["Baseline", "T=2, d=0.3", "T=4, d=0.3", "Stride=1, d=0.3"]
    matrix = []
    for m in hmap_models:
        row = [x if x is not None else np.nan for x in data[m]]
        matrix.append(row)

    arr = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(arr, ax=ax, annot=True, fmt=".1f", cmap=cmap,
                xticklabels=ROUTE_LABELS, yticklabels=hmap_models,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Driving Score", "shrink": 0.8},
                vmin=0, vmax=25)

    ax.set_title("Per-Route Driving Score Heatmap — Driving Models (dropout=0.3)\n"
                 "(Town05 routes S16–S25; NaN = eval still running)", fontsize=11)
    ax.set_xlabel("Route", fontsize=11)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    _save(fig, "fig2_per_route_heatmap")

# ── Plot 3: Dropout ablation (d=0.1 vs d=0.3 side-by-side) ───────────────────
def plot_dropout_ablation():
    pairs = [
        ("T=2", "T=2, d=0.1", "T=2, d=0.3"),
        ("T=4", "T=4, d=0.1", "T=4, d=0.3"),
        ("Stride=1", "Stride=1, d=0.1", "Stride=1, d=0.3"),
        ("CrossAttn", "CrossAttn, d=0.1", None),
    ]

    x = np.arange(len(pairs))
    w = 0.32
    d01_vals = [AVGS[p[1]] for p in pairs]
    d03_vals = [AVGS[p[2]] if p[2] else 0.0 for p in pairs]
    labels = [p[0] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, d01_vals, w, label="dropout=0.1", color=RED,   edgecolor="white")
    b2 = ax.bar(x + w/2, d03_vals, w, label="dropout=0.3", color=GREEN, edgecolor="white")

    for bar, val in zip(list(b1) + list(b2), d01_vals + d03_vals):
        if val > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(AVGS["Baseline"], color=GRAY, linestyle="--", linewidth=1.3,
               label=f"Baseline ({AVGS['Baseline']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Avg Driving Score (DS)", fontsize=12)
    ax.set_title("Dropout Ablation: d=0.1 vs d=0.3\n"
                 "(All d=0.1 models produce DS=0.0 — vehicle never moves)", fontsize=11)
    ax.set_ylim(0, max(d03_vals) * 1.25)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "fig3_dropout_ablation")

# ── Plot 4: Temporal frames scaling (d=0.3 only) ──────────────────────────────
def plot_t_scaling():
    t_vals = [1, 2, 4]
    ds_vals = [AVGS["Baseline"], AVGS["T=2, d=0.3"], AVGS["T=4, d=0.3"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_vals, ds_vals, "o-", color=BLUE, linewidth=2.2, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5)

    for x, y in zip(t_vals, ds_vals):
        ax.annotate(f"DS={y:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, 8), fontsize=11, fontweight="bold", color=BLUE)

    ax.set_xticks(t_vals)
    ax.set_xticklabels(["T=1\n(Baseline)", "T=2", "T=4"])
    ax.set_xlabel("Number of Temporal Frames (T)", fontsize=12)
    ax.set_ylabel("Avg Driving Score (DS)", fontsize=12)
    ax.set_title("Effect of Temporal Horizon on Driving Score\n"
                 "(dropout=0.3, stride=5; T=1 is InterFuser baseline)", fontsize=11)
    ax.set_ylim(0, max(ds_vals) * 1.25)
    ax.fill_between(t_vals, ds_vals, alpha=0.08, color=BLUE)
    plt.tight_layout()
    _save(fig, "fig4_t_scaling")

# ── Plot 5: Stride comparison (stride=1 vs stride=5, d=0.3) ──────────────────
def plot_stride_comparison():
    models = ["Baseline\n(T=1)", "Stride=5\n(T=4, d=0.3)", "Stride=1\n(T=4, d=0.3)"]
    vals   = [AVGS["Baseline"], AVGS["T=4, d=0.3"], AVGS["Stride=1, d=0.3"]]
    colors = [GRAY, GREEN, PURPLE]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.45)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(AVGS["Baseline"], color=GRAY, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_ylabel("Avg Driving Score (DS)", fontsize=12)
    ax.set_title("Temporal Stride Comparison (T=4, dropout=0.3)\n"
                 "Stride=5: 2-second history window; Stride=1: dense consecutive frames", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)

    note = "* Stride=1 d=0.3 uses 8/10 routes\n  (batch 4 eval pending)"
    ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="gray", style="italic")
    plt.tight_layout()
    _save(fig, "fig5_stride_comparison")

# ── Plot 6: Route-by-route grouped bars (d=0.3 driving models) ───────────────
def plot_route_grouped():
    driving_models = ["Baseline", "T=2, d=0.3", "T=4, d=0.3", "Stride=1, d=0.3"]
    colors_m = [GRAY, ORANGE, GREEN, PURPLE]
    n_models = len(driving_models)
    n_routes = 10
    x = np.arange(n_routes)
    total_w = 0.75
    w = total_w / n_models

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, col) in enumerate(zip(driving_models, colors_m)):
        vals = [x if x is not None else 0.0 for x in data[m]]
        offset = (i - n_models/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=m, color=col, edgecolor="white", linewidth=0.5, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(ROUTE_LABELS, fontsize=11)
    ax.set_xlabel("Route", fontsize=12)
    ax.set_ylabel("Driving Score (DS)", fontsize=12)
    ax.set_title("Per-Route Driving Score — Driving Models Comparison\n"
                 "(dropout=0.3 only; all dropout=0.1 models DS=0.0 omitted)", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    _save(fig, "fig6_route_grouped_bars")

# ── Plot 7: Summary comparison (3-panel) ─────────────────────────────────────
def plot_summary_panel():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Avg DS driving models only
    ax = axes[0]
    models_a = ["Baseline", "T=2\nd=0.3", "T=4\nd=0.3", "Stride=1\nd=0.3"]
    keys_a   = ["Baseline", "T=2, d=0.3", "T=4, d=0.3", "Stride=1, d=0.3"]
    vals_a   = [AVGS[k] for k in keys_a]
    cols_a   = [GRAY, ORANGE, GREEN, PURPLE]
    bars_a = ax.bar(models_a, vals_a, color=cols_a, edgecolor="white", width=0.55)
    for bar, val in zip(bars_a, vals_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(AVGS["Baseline"], color=GRAY, linestyle="--", linewidth=1.1, alpha=0.6)
    ax.set_title("A. Avg DS (driving models)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg DS")
    ax.set_ylim(0, 11)

    # Panel B: T scaling
    ax = axes[1]
    t_x = [1, 2, 4]
    t_y = [AVGS["Baseline"], AVGS["T=2, d=0.3"], AVGS["T=4, d=0.3"]]
    ax.plot(t_x, t_y, "o-", color=BLUE, lw=2.2, ms=9,
            markerfacecolor="white", markeredgewidth=2.5)
    for xv, yv in zip(t_x, t_y):
        ax.annotate(f"{yv:.2f}", (xv, yv), xytext=(5, 7),
                    textcoords="offset points", fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xticks(t_x)
    ax.set_xticklabels(["T=1\n(base)", "T=2", "T=4"])
    ax.set_title("B. Temporal Horizon Scaling (d=0.3)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg DS")
    ax.set_ylim(0, 11)
    ax.fill_between(t_x, t_y, alpha=0.08, color=BLUE)

    # Panel C: % improvement vs baseline
    ax = axes[2]
    models_c = ["T=2\nd=0.3", "T=4\nd=0.3", "Stride=1\nd=0.3"]
    keys_c   = ["T=2, d=0.3", "T=4, d=0.3", "Stride=1, d=0.3"]
    base_ds  = AVGS["Baseline"]
    pcts = [(AVGS[k] - base_ds) / base_ds * 100 for k in keys_c]
    bar_colors = [GREEN if p > 0 else RED for p in pcts]
    bars_c = ax.bar(models_c, pcts, color=bar_colors, edgecolor="white", width=0.45)
    for bar, val in zip(bars_c, pcts):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2.5
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("C. % Improvement vs Baseline", fontsize=11, fontweight="bold")
    ax.set_ylabel("DS Change (%)")

    fig.suptitle("Temporal InterFuser — CARLA Closed-Loop Evaluation Summary\n"
                 "(Town05, 10 routes; dropout=0.3 required for vehicle to move)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig7_summary_panel")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating thesis plots...")
    plot_avg_ds()
    plot_heatmap()
    plot_dropout_ablation()
    plot_t_scaling()
    plot_stride_comparison()
    plot_route_grouped()
    plot_summary_panel()
    print(f"\nAll figures saved to: {os.path.abspath(FIGURES_DIR)}")
