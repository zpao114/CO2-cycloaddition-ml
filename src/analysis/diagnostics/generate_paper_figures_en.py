# -*- coding: utf-8 -*-
"""
generate_paper_figures_en.py - Paper Figure Generation Script (English v5)
===================================================================
Generate 8 high-quality figures for the CO2-cycloaddition paper (English version).

Improvements over v4:
- Fig 0: Optimized layout, larger pie chart
- Fig 1: Larger panels, optimized fonts
- Fig 2: Fixed protocol name display
- Fig 3: R² display optimization, layout adjustment
- Fig 4: Colorbar number format fixed
- Fig 5: Legend layout optimization
- Fig 6: Optimal λ annotation position fixed
- Fig 7: Bar chart number annotation spacing fixed

Usage:
    python src/analysis/diagnostics/generate_paper_figures_en.py
"""

from __future__ import annotations

import os
import sys
import io
import warnings
from datetime import datetime

# UTF-8 output support
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_absolute_error

# svglib is not used for SVG conversion
SVGLIB_AVAILABLE = False

# Use PyMuPDF for PDF->SVG conversion
try:
    import fitz
    MUPDF_AVAILABLE = True
except ImportError:
    MUPDF_AVAILABLE = False

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
os.chdir(PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Data file paths
PATH_SUMMARY_PROTOCOL = os.path.join(PROJECT_ROOT, "results_step4", "summary_protocol.csv")
PATH_LOSO_PREDICTIONS = os.path.join(PROJECT_ROOT, "results_step7_improved_loso", "loso_predictions.csv")
PATH_SHAP_FEATURES = os.path.join(PROJECT_ROOT, "results_step4_5", "per_substrate_top_features.csv")
PATH_LATENT = os.path.join(PROJECT_ROOT, "results_pcl_ae_viz", "latent_comparison.csv")
PATH_LAMBDA = os.path.join(PROJECT_ROOT, "results_lambda_ablation", "lambda_results.csv")
PATH_LEARNING_CURVE = os.path.join(PROJECT_ROOT, "results_sample_size_sensitivity", "learning_curve_summary.csv")

# ============================================================================
# Font Configuration
# ============================================================================
def setup_font():
    """Configure font for matplotlib"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("  Using font: DejaVu Sans")

setup_font()
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# Global Plot Configuration
# ============================================================================
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

# ============================================================================
# Constants
# ============================================================================
SUB2 = "\u2082"   # CO₂
MINUS = "\u2212"  # −

COLOR_CHO = "#d62728"
COLOR_TERM = "#2ca02c"
COLOR_OTH = "#1f77b4"

SUB_COLORS = {
    "Cyclohexene oxide": COLOR_CHO,
    "Propylene oxide": COLOR_TERM,
    "Epichlorohydrin": COLOR_OTH,
    "Styrene oxide": "#ff7f0e",
    "Isopropyl glycidyl ether": "#9467bd",
}

SUB_SHORT = {
    "Cyclohexene oxide": "CHO",
    "Propylene oxide": "PO",
    "Epichlorohydrin": "ECH",
    "Styrene oxide": "SO",
    "Isopropyl glycidyl ether": "IGE",
}

SUBSTRATES_ORDER = [
    "Cyclohexene oxide",
    "Styrene oxide",
    "Propylene oxide",
    "Epichlorohydrin",
    "Isopropyl glycidyl ether",
]

# ============================================================================
# Helper Functions
# ============================================================================
def save_figure(fig, name_base: str):
    """Save figure as PNG, PDF, and SVG"""
    png_path = os.path.join(FIGURES_DIR, f"{name_base}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name_base}.pdf")
    fig.savefig(png_path, dpi=300, facecolor="white", pad_inches=0.1)
    fig.savefig(pdf_path, dpi=300, facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    # SVG conversion via PyMuPDF
    if MUPDF_AVAILABLE:
        try:
            svg_path = os.path.join(FIGURES_DIR, f"{name_base}.svg")
            doc = fitz.open(pdf_path)
            page = doc[0]
            svg_data = page.get_svg_image()
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_data)
            doc.close()
            print(f"  Saved: {svg_path}")
        except Exception as e:
            print(f"  SVG conversion skipped: {e}")


def format_r2(value: float) -> str:
    """Format R² value"""
    if value < 0:
        return f"{MINUS}{abs(value):.3f}"
    return f"{value:.3f}"


# ============================================================================
# Fig 0: Graphical Abstract - Optimized
# ============================================================================
def make_fig0_graphical_abstract():
    """Generate graphical abstract with optimized layout"""

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.2, 1, 1, 1.5], wspace=0.3)

    # ----- Panel (a): Data cleaning funnel -----
    ax = fig.add_subplot(gs[0])

    funnel = [5263, 3733, 2602, 2316]
    funnel_labels = ["5,263 (raw)", "3,733", "2,602", "2,316 (final)"]
    funnel_colors = ["#90A4AE", "#78909C", "#607D8B", "#455A64"]

    bars = ax.barh(range(len(funnel)), funnel, color=funnel_colors, edgecolor="black", height=0.6)
    for i, (v, lbl) in enumerate(zip(funnel, funnel_labels)):
        ax.text(v + 100, i, lbl, va="center", ha="left", fontsize=10)

    ax.set_yticks([])
    ax.set_xlim(0, 6200)
    ax.set_xlabel("Number of reactions", fontsize=11)
    ax.set_title("(a) Data Cleaning Funnel", fontweight="bold", fontsize=12)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Panel (b): Catalyst system distribution -----
    ax = fig.add_subplot(gs[1])

    pie_sizes = [79.6, 7.6, 6.7, 2.8, 3.2]
    pie_labels = ["IL 79.6%", "MH 7.6%", "Mixed 6.7%", "BAS 2.8%", "Unk 3.2%"]
    pie_colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#9E9E9E"]

    wedges, texts, autotexts = ax.pie(pie_sizes, colors=pie_colors, startangle=90,
                                       wedgeprops=dict(edgecolor="white", linewidth=2),
                                       autopct="")
    ax.legend(wedges, pie_labels, loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=9)
    ax.set_title("(b) Catalyst Distribution", fontweight="bold", fontsize=12)

    # ----- Panel (c): Substrate yield comparison -----
    ax = fig.add_subplot(gs[2])

    substrates_yield = ["CHO", "SO", "PO", "ECH", "IGE"]
    mean_yields = [53.1, 84.4, 88.3, 91.9, 91.0]
    counts = [305, 783, 646, 692, 64]
    colors_bar = [COLOR_CHO, "#ff7f0e", COLOR_TERM, COLOR_OTH, "#9467bd"]

    bars = ax.bar(substrates_yield, mean_yields, color=colors_bar, edgecolor="black", width=0.65)
    ax.set_ylabel("Mean Yield (%)", fontsize=11)
    ax.set_xlabel("Substrate", fontsize=11)
    ax.set_title("(c) Substrate Yield", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, y, n in zip(bars, mean_yields, counts):
        ax.text(bar.get_x() + bar.get_width()/2, y + 2, f"{y:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ----- Panel (d): Key findings -----
    ax = fig.add_subplot(gs[3])
    ax.axis("off")

    findings = [
        r"$\mathbf{Key\ Findings}$",
        "",
        r"1. GroupKFold $R^2 = 0.278 \rightarrow LOSO\ R^2 = -0.063$",
        r"   $\rightarrow$ Model fails on unseen substrates",
        "",
        r"2. CHO dominates LOSO failure:",
        r"   $\bullet$ Mean bias +34.6%",
        r"   $\bullet$ sub_homo_eV: $r = -0.245$",
        "",
        r"3. PCL-AE better separates catalyst",
        r"   families (score: 0.81 vs 0.54)",
    ]

    text = "\n".join(findings)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#333333", linewidth=1.5))

    fig.suptitle(r"Graphical Abstract — CO$_2$ Cycloaddition ML Framework",
                 fontsize=15, fontweight="bold", y=0.98)

    save_figure(fig, "fig0_graphical_abstract_en")
    print("  Fig 0 done")


# ============================================================================
# Fig 1: Data Overview - Optimized
# ============================================================================
def make_fig1_data_overview():
    """Generate data overview with larger panels"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)
    df_pred["predicted"] = df_pred["predicted"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ----- Panel (a): Sample size distribution -----
    ax = axes[0]
    substrates = SUBSTRATES_ORDER
    counts = [305, 783, 646, 692, 64]
    colors = [SUB_COLORS.get(s, COLOR_OTH) for s in substrates]

    bars = ax.bar([SUB_SHORT[s] for s in substrates], counts, color=colors, edgecolor="black", width=0.6)
    ax.set_ylabel("Sample Count", fontsize=14)
    ax.set_xlabel("Substrate", fontsize=14)
    ax.set_title("(a) Sample Size Distribution", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"n={n}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # ----- Panel (b): Yield distribution boxplot -----
    ax = axes[1]
    box_data = []
    box_positions = []

    for i, sub in enumerate(substrates):
        d = df_pred[df_pred["substrate"] == sub]["actual"].values
        box_data.append(d)
        box_positions.append(i)

    bp = ax.boxplot(box_data, positions=box_positions, widths=0.5,
                    patch_artist=True, showfliers=True, flierprops=dict(markersize=4))

    for patch, sub in zip(bp["boxes"], substrates):
        patch.set_facecolor(SUB_COLORS.get(sub, COLOR_OTH))
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=11)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.set_xlabel("Substrate", fontsize=12)
    ax.set_title("(b) Yield Distribution", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-5, 110)

    # ----- Panel (c): Prediction error distribution -----
    ax = axes[2]

    for i, sub in enumerate(substrates):
        d = df_pred[df_pred["substrate"] == sub]["error"].values
        ax.hist(d, bins=25, alpha=0.5, label=SUB_SHORT[sub],
                color=SUB_COLORS.get(sub, COLOR_OTH), density=True)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction Error (actual - predicted, %)", fontsize=11)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("(c) Error Distribution", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)

    plt.tight_layout()
    save_figure(fig, "fig1_data_overview_en")
    print("  Fig 1 done")


# ============================================================================
# Fig 2: Protocol Comparison - Fixed
# ============================================================================
def make_fig2_protocol_comparison():
    """Generate CV protocol comparison with fixed protocol names"""

    df = pd.read_csv(PATH_SUMMARY_PROTOCOL, encoding="utf-8-sig")
    df["r2"] = df["r2"].astype(float)
    df["mae"] = df["mae"].astype(float)

    # Use "LOSO x LOMO" instead of "LOSO×LOMO" to avoid encoding issues
    protocols = ["5fold", "LOSO", "LOMO", "LOSO x LOMO"]
    x0 = df[df["feature_set"] == "X0_xTB_only"].set_index("protocol")
    x1 = df[df["feature_set"] == "X1_xTB+mech"].set_index("protocol")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Panel (a): R² comparison -----
    ax = axes[0]
    x = np.arange(len(protocols))
    w = 0.35

    r2_x0 = [x0.loc[p, "r2"] if p in x0.index else 0 for p in protocols]
    r2_x1 = [x1.loc[p, "r2"] if p in x1.index else 0 for p in protocols]

    bars1 = ax.bar(x - w/2, r2_x0, w, label="X0 = xTB only", color="#64B5F6", edgecolor="black")
    bars2 = ax.bar(x + w/2, r2_x1, w, label="X1 = xTB + mech", color="#1976D2", edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=10)
    ax.set_ylabel("R\u00b2", fontsize=12)
    ax.set_title("Protocol R\u00b2 Comparison (XGBoost)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, (v0, v1) in enumerate(zip(r2_x0, r2_x1)):
        offset = 0.02 if v0 >= 0 else -0.08
        ax.text(i - w/2, v0 + offset, f"{v0:.3f}", ha="center", fontsize=9, fontweight="bold")
        offset = 0.02 if v1 >= 0 else -0.08
        ax.text(i + w/2, v1 + offset, f"{v1:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(-0.5, 0.6)

    # ----- Panel (b): MAE comparison -----
    ax = axes[1]

    mae_x0 = [x0.loc[p, "mae"] if p in x0.index else 0 for p in protocols]
    mae_x1 = [x1.loc[p, "mae"] if p in x1.index else 0 for p in protocols]

    ax.bar(x - w/2, mae_x0, w, label="X0 = xTB only", color="#90CAF9", edgecolor="black")
    ax.bar(x + w/2, mae_x1, w, label="X1 = xTB + mech", color="#1565C0", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=10)
    ax.set_ylabel("MAE (%)", fontsize=12)
    ax.set_title("Protocol MAE Comparison (XGBoost)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, (v0, v1) in enumerate(zip(mae_x0, mae_x1)):
        ax.text(i - w/2, v0 + 0.3, f"{v0:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w/2, v1 + 0.3, f"{v1:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "fig2_protocol_comparison_en")
    print("  Fig 2 done")


# ============================================================================
# Fig 3: LOSO Root Cause - Optimized
# ============================================================================
def make_fig3_loso_root_cause():
    """Generate LOSO root cause analysis with optimized layout"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)
    df_pred["predicted"] = df_pred["predicted"].astype(float)
    df_pred["error"] = df_pred["actual"] - df_pred["predicted"]

    substrates = SUBSTRATES_ORDER
    CHO = "Cyclohexene oxide"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ----- Panel (A): Scatter plot -----
    ax = axes[0, 0]
    for sub in substrates:
        d = df_pred[df_pred["substrate"] == sub]
        ax.scatter(d["predicted"], d["actual"], alpha=0.4, s=20,
                   color=SUB_COLORS.get(sub, COLOR_OTH),
                   label=f"{SUB_SHORT[sub]} (n={len(d)})")

    ax.plot([0, 100], [0, 100], "k--", alpha=0.6, lw=2, label="y = x")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Predicted Yield (%)", fontsize=12)
    ax.set_ylabel("Actual Yield (%)", fontsize=12)
    ax.set_title("(A) Predicted vs Actual", fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    r2_all = r2_score(df_pred["actual"], df_pred["predicted"])
    mae_all = mean_absolute_error(df_pred["actual"], df_pred["predicted"])
    ax.text(0.97, 0.03, f"R\u00b2 = {r2_all:.4f}\nMAE = {mae_all:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    # ----- Panel (B): Bias bar chart -----
    ax = axes[0, 1]

    bias_data = []
    for sub in substrates:
        d = df_pred[df_pred["substrate"] == sub]
        bias_data.append({
            "sub": SUB_SHORT[sub],
            "bias": d["actual"].mean() - d["predicted"].mean(),
            "r2": r2_score(d["actual"], d["predicted"]),
            "n": len(d),
        })

    x_pos = np.arange(len(bias_data))
    colors_bar = [COLOR_CHO if SUB_SHORT[s] == "CHO" else SUB_COLORS.get(s, COLOR_OTH)
                  for s in substrates]

    bars = ax.bar(x_pos, [b["bias"] for b in bias_data], color=colors_bar, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["sub"] for b in bias_data], fontsize=11)
    ax.set_ylabel("Bias (Actual - Predicted, %)", fontsize=11)
    ax.set_title("(B) Prediction Bias by Substrate", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for i, b in enumerate(bias_data):
        h = b["bias"]
        va = "bottom" if h >= 0 else "top"
        ax.text(i, h + (3 if h >= 0 else -3), f"{h:+.1f}%", ha="center", va=va,
                fontsize=10, fontweight="bold")

    ax.set_ylim(-50, 50)

    # ----- Panel (C): Box plot -----
    ax = axes[1, 0]

    box_d, positions, colors_list = [], [], []
    pos = 0
    for sub in substrates:
        d_actual = df_pred[df_pred["substrate"] == sub]["actual"].values
        d_predicted = df_pred[df_pred["substrate"] == sub]["predicted"].values
        box_d.append(d_actual)
        positions.append(pos)
        pos += 1
        box_d.append(d_predicted)
        positions.append(pos)
        pos += 2
        colors_list.extend([SUB_COLORS.get(sub, COLOR_OTH)] * 2)

    bp = ax.boxplot(box_d, positions=positions, widths=0.6, patch_artist=True, showfliers=False)

    for pi, (patch, col) in enumerate(zip(bp["boxes"], colors_list)):
        patch.set_facecolor(col)
        patch.set_alpha(0.5 if pi % 2 == 0 else 0.9)
        if pi % 2 == 1:
            patch.set_hatch("//")

    sub_centers = [positions[2*i] + 0.5 for i in range(len(substrates))]
    ax.set_xticks(sub_centers)
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=11)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.set_title("(C) Actual (light) vs Predicted (hatched)", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.5, label="Actual"),
        mpatches.Patch(facecolor="gray", alpha=0.9, hatch="//", label="Predicted"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_ylim(-5, 105)

    # ----- Panel (D): Heatmap -----
    ax = axes[1, 1]

    pivot_err = df_pred.pivot_table(index="mechanism", columns="substrate",
                                    values="error", aggfunc="mean")
    pivot_n = df_pred.pivot_table(index="mechanism", columns="substrate",
                                  values="error", aggfunc="count")

    mech_order = ["ionic_liquid", "metal_halide", "mixed_system", "organic_base", "unknown"]
    pivot_err = pivot_err.reindex(index=mech_order, columns=SUBSTRATES_ORDER)
    pivot_n = pivot_n.reindex(index=mech_order, columns=SUBSTRATES_ORDER, fill_value=0)

    vmax = max(abs(pivot_err.values.max()), abs(pivot_err.values.min())) * 1.2
    im = ax.imshow(pivot_err.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(SUBSTRATES_ORDER)))
    ax.set_xticklabels([SUB_SHORT[s] for s in SUBSTRATES_ORDER], fontsize=11)
    ax.set_yticks(range(len(mech_order)))
    ax.set_yticklabels(mech_order, fontsize=10)

    for i in range(len(mech_order)):
        for j in range(len(SUBSTRATES_ORDER)):
            v = pivot_err.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", color=color, fontsize=11)

    plt.colorbar(im, ax=ax, label="Bias: Actual - Predicted (%)", shrink=0.9)
    ax.set_title("(D) Bias Heatmap (Substrate x Mechanism)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Substrate (held out)", fontsize=12)

    fig.suptitle("LOSO Root Cause: CHO Structural Bias", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig3_loso_root_cause_en")
    print("  Fig 3 done")


# ============================================================================
# Fig 4: Coverage Matrix - Fixed Colorbar
# ============================================================================
def make_fig4_coverage_matrix():
    """Generate 5x5 coverage matrix with fixed colorbar format"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)

    substrates = SUBSTRATES_ORDER
    mechanisms = ["ionic_liquid", "metal_halide", "mixed_system", "organic_base", "unknown"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ----- Panel (a): Sample count -----
    ax = axes[0]

    n_matrix = np.zeros((len(mechanisms), len(substrates)), dtype=int)
    for i, mech in enumerate(mechanisms):
        for j, sub in enumerate(substrates):
            n_matrix[i, j] = len(df_pred[(df_pred["mechanism"] == mech) & (df_pred["substrate"] == sub)])

    im0 = ax.imshow(n_matrix, cmap="Blues", aspect="auto", norm=LogNorm(vmin=1, vmax=n_matrix.max()))
    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=12)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms, fontsize=11)

    for i in range(len(mechanisms)):
        for j in range(len(substrates)):
            v = int(n_matrix[i, j])
            color = "white" if v > 200 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    cbar0 = plt.colorbar(im0, ax=ax, shrink=0.8)
    cbar0.ax.set_ylabel("Sample Count", fontsize=11)

    ax.set_title("(a) Sample Coverage Matrix (n)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Substrate", fontsize=12)
    ax.set_ylabel("Catalyst Mechanism", fontsize=12)

    # ----- Panel (b): Mean yield -----
    ax = axes[1]

    yield_matrix = np.full((len(mechanisms), len(substrates)), np.nan)
    for i, mech in enumerate(mechanisms):
        for j, sub in enumerate(substrates):
            d = df_pred[(df_pred["mechanism"] == mech) & (df_pred["substrate"] == sub)]
            if len(d) > 0:
                yield_matrix[i, j] = d["actual"].mean()

    im1 = ax.imshow(yield_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=12)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms, fontsize=11)

    for i in range(len(mechanisms)):
        for j in range(len(substrates)):
            v = yield_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
    cbar1.ax.set_ylabel("Mean Yield (%)", fontsize=11)

    ax.set_title("(b) Mean Yield Matrix (%)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Substrate", fontsize=12)
    ax.set_ylabel("Catalyst Mechanism", fontsize=12)

    fig.suptitle("5x5 Substrate x Catalyst Coverage Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig4_coverage_matrix_en")
    print("  Fig 4 done")


# ============================================================================
# Fig 5: SHAP Direction - Optimized
# ============================================================================
def make_fig5_shap_direction():
    """Generate SHAP directionality analysis with optimized layout"""

    df_shap = pd.read_csv(PATH_SHAP_FEATURES, encoding="utf-8-sig")
    df_shap["mean_signed_shap"] = df_shap["mean_signed_shap"].astype(float)
    df_shap["mean_abs_shap"] = df_shap["mean_abs_shap"].astype(float)

    substrates = SUBSTRATES_ORDER
    CHO = "Cyclohexene oxide"

    top_feats = df_shap.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ----- Panel (a): Grouped bar chart -----
    ax = axes[0]
    n_feats = len(top_feats)
    width = 0.14
    x = np.arange(n_feats)

    for i, sub in enumerate(substrates):
        sub_data = df_shap[df_shap["substrate"] == sub].set_index("feature")
        vals = [sub_data.loc[f, "mean_signed_shap"] if f in sub_data.index else 0 for f in top_feats]
        color = COLOR_CHO if sub == CHO else SUB_COLORS.get(sub, COLOR_OTH)
        ax.bar(x + (i - 2) * width, vals, width, label=SUB_SHORT.get(sub, sub),
               color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean Signed SHAP Value", fontsize=12)
    ax.set_title("(a) Signed SHAP by Substrate", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ----- Panel (b): Heatmap -----
    ax = axes[1]

    plot_data = df_shap[df_shap["feature"].isin(top_feats)]
    pivot = plot_data.pivot(index="feature", columns="substrate", values="mean_signed_shap")
    pivot = pivot.reindex(columns=substrates, fill_value=0)
    pivot = pivot.loc[top_feats]

    vmax = max(abs(pivot.values.max()), abs(pivot.values.min())) * 1.2
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-max(vmax, 5), vmax=max(vmax, 5))

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT.get(s, s) for s in substrates], fontsize=12)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=10)

    for i in range(len(top_feats)):
        for j in range(len(substrates)):
            v = pivot.values[i, j]
            color = "white" if abs(v) > max(vmax, 5) * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Mean Signed SHAP", shrink=0.85)
    ax.set_title("(b) SHAP Direction Heatmap", fontweight="bold", fontsize=13)
    ax.set_xlabel("Substrate", fontsize=12)

    fig.suptitle("Per-Substrate SHAP Directionality", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig5_shap_direction_en")
    print("  Fig 5 done")


# ============================================================================
# Fig 6: Ablation - Optimized
# ============================================================================
def make_fig6_ablation():
    """Generate ablation figure with optimized annotation"""

    df_lambda = pd.read_csv(PATH_LAMBDA, encoding="utf-8-sig")
    df_lc = pd.read_csv(PATH_LEARNING_CURVE, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ----- Panel (a): Lambda ablation -----
    ax = axes[0]

    lambdas = df_lambda["lambda"].values
    ann_r2 = df_lambda["ann_r2_mean"].values
    rf_r2 = df_lambda["rf_r2_mean"].values

    ax.plot(lambdas, ann_r2, "o-", color="#1976D2", label="DualBranchANN", linewidth=2.5, markersize=7)
    ax.plot(lambdas, rf_r2, "s-", color="#2ca02c", label="RandomForest", linewidth=2.5, markersize=7)

    # Mark best point
    best_idx = df_lambda["ann_r2_mean"].idxmax()
    best_lambda = df_lambda.loc[best_idx, "lambda"]
    best_r2 = df_lambda.loc[best_idx, "ann_r2_mean"]

    ax.axvline(best_lambda, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.scatter([best_lambda], [best_r2], color="red", s=150, zorder=5, marker="*")
    ax.annotate(f"\u03bb = {best_lambda}\nR\u00b2 = {best_r2:.3f}",
                xy=(best_lambda, best_r2),
                xytext=(best_lambda * 3, best_r2 - 0.03),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("\u03bb (KL Divergence Weight)", fontsize=12)
    ax.set_ylabel("R\u00b2", fontsize=12)
    ax.set_title("(a) KL Divergence Weight Ablation", fontweight="bold", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # ----- Panel (b): Sample size sensitivity -----
    ax = axes[1]

    for model, color, marker in [("RF", "#e53935", "o"), ("LGBM", "#fb8c00", "s")]:
        data = df_lc[df_lc["model"] == model].sort_values("n_train")
        ax.plot(data["n_train"], data["r2_mean"], f"{marker}-",
                color=color, label=model, linewidth=2.5, markersize=7)
        ax.fill_between(data["n_train"],
                        data["r2_mean"] - data["r2_std"],
                        data["r2_mean"] + data["r2_std"],
                        alpha=0.2, color=color)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Training Sample Size (n)", fontsize=12)
    ax.set_ylabel("R\u00b2", fontsize=12)
    ax.set_title("(b) Sample Size Sensitivity", fontweight="bold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    save_figure(fig, "fig6_ablation_en")
    print("  Fig 6 done")


# ============================================================================
# Fig 7: Latent Space - Optimized
# ============================================================================
def make_fig7_latent_space():
    """Generate latent space comparison with optimized layout"""

    df_latent = pd.read_csv(PATH_LATENT, encoding="utf-8-sig")

    models = df_latent["model"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ----- Panel (a): Feature-related dimensions -----
    ax = axes[0]
    x = np.arange(len(models))
    n_dims_r = df_latent["n_dims_with_|r|>0.1"].values
    n_dims_total = df_latent["n_dims"].values

    bars = ax.bar(x, n_dims_r, color=["#1976D2", "#e53935"], edgecolor="black", width=0.5)
    ax.bar(x, n_dims_total - n_dims_r, bottom=n_dims_r, color=["#90CAF9", "#FFCDD2"], edgecolor="black", width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Standard AE", "PCL-AE"], fontsize=12)
    ax.set_ylabel("Number of Dimensions", fontsize=12)
    ax.set_title("(a) Feature-Related Dimensions", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, n_r, n_total in zip(bars, n_dims_r, n_dims_total):
        ax.text(bar.get_x() + bar.get_width()/2, n_r/2,
                f"{int(n_r)}/{int(n_total)}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")

    ax.legend([mpatches.Patch(color="#90CAF9"), mpatches.Patch(color="#1976D2")],
              ["Uncorrelated", "Correlated (|r|>0.1)"], loc="upper left", fontsize=9)

    # ----- Panel (b): Quality metrics -----
    ax = axes[1]

    fractions = df_latent["fraction_x"].values
    mean_rs = df_latent["mean_|r|"].values

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, fractions, width, label="fraction_x", color="#1976D2", edgecolor="black")
    ax.bar(x + width/2, mean_rs, width, label="mean_|r|", color="#2ca02c", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Standard AE", "PCL-AE"], fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("(b) Latent Space Quality", fontweight="bold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (f, r) in enumerate(zip(fractions, mean_rs)):
        ax.text(i - width/2, f + 0.03, f"{f:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + width/2, r + 0.03, f"{r:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.0)

    # ----- Panel (c): Yield correlation & clustering -----
    ax = axes[2]

    rho_yield = df_latent["rho_yield"].values
    silhouette = df_latent["silhouette_catalyst_family"].values

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, rho_yield, width, label="\u03c1(yield)", color="#ff9800", edgecolor="black")
    ax.bar(x + width/2, silhouette, width, label="silhouette", color="#9c27b0", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Standard AE", "PCL-AE"], fontsize=12)
    ax.set_ylabel("Coefficient", fontsize=12)
    ax.set_title("(c) Yield Correlation & Clustering", fontweight="bold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    for i, (r, s) in enumerate(zip(rho_yield, silhouette)):
        ax.text(i - width/2, r + 0.02, f"{r:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + width/2, s + 0.02, f"{s:.3f}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("PCL-AE vs Standard AE Latent Space", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig7_latent_space_en")
    print("  Fig 7 done")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Generate all 8 paper figures (English version)"""

    print("=" * 70)
    print("Generating Paper Figures v5 (English)")
    print("=" * 70)
    print(f"Output: {FIGURES_DIR}")
    print()

    start_time = datetime.now()

    make_fig0_graphical_abstract()
    make_fig1_data_overview()
    make_fig2_protocol_comparison()
    make_fig3_loso_root_cause()
    make_fig4_coverage_matrix()
    make_fig5_shap_direction()
    make_fig6_ablation()
    make_fig7_latent_space()

    elapsed = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 70)
    print(f"Done! Generated 16 figures (8 PNG + 8 PDF) in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
