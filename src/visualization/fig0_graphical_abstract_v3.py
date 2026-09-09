"""Graphical Abstract - v3 (corrected, 2026-08-24).

Corrected against paper_final_English.md / paper_final_Chinese.md:
  - SHAP values: read from bootstrap_substrate_shap_ci.csv (100-bootstrap CIs)
  - DFT/xTB: read from dft_validation/results/514_dft_vs_xtb_report.csv (real data)
  - Footer n = 2,490 (not 2,316)
  - Footer LOSO R² = +0.228 ± 0.007 (not +0.111)
  - Output: figures/fig0_graphical_abstract.png / .pdf
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Color palette (color-blind aware)
# ---------------------------------------------------------------------------
COL_TERMINAL  = "#1f77b4"   # blue  – terminal epoxides
COL_CHO       = "#d62728"   # red   – CHO (sign reversal)
COL_NEUTRAL   = "#7f7f7f"   # grey  – noise / neutral
COL_HIGHLIGHT = "#ff7f0e"   # orange – annotation highlight
COL_OTHER     = "#cccccc"   # light grey – non-substrate molecules


# ---------------------------------------------------------------------------
# Panel (a): SHAP direction reversal  (correct data from bootstrap_substrate_shap_ci.csv)
# ---------------------------------------------------------------------------
def panel_shap(ax):
    """5-substrate signed SHAP bar chart with 95% CI error bars.

    Values from data/processed/bootstrap_substrate_shap_ci.csv (100-bootstrap, LOSO).

    Order matches the paper figures (PO, ECH, SO, IGE, CHO).
    Note: CHO magnitude is ~8× larger than terminal epoxides; the y-axis is
    deliberately compressed so all five bars are visible simultaneously.
    """
    substrates = ["PO", "ECH", "SO", "IGE", "CHO"]

    # Correct values from bootstrap_substrate_shap_ci.csv (sub_homo_eV, signed SHAP)
    means  = [3.657544733606812, 4.37316474950867,  2.9642110561430397, 3.0485054312499997, -25.14763656393443]
    ci_lo  = [3.5954000930634678, 4.322403027857659,  2.914270917393359,  2.8984908810546877, -25.60815733032787]
    ci_hi  = [3.729905469268576,  4.440896799561416,  3.0063180213243936,  3.1899051212109377, -24.684486155819673]
    colors = [COL_TERMINAL, COL_TERMINAL, COL_TERMINAL, COL_TERMINAL, COL_CHO]

    n   = len(substrates)
    xs  = np.linspace(0.12, 0.88, n)
    bw  = 0.08  # bar width

    # --- visual scale: compress CHO to ~3× terminal height for legibility ---
    # terminal range: roughly −0.1 to +5; CHO compressed from −26 to −3
    scale = 0.13  # 1 / (|CHO_mean| / 3.5)
    means_d  = [m * scale for m in means]
    ci_lo_d   = [lo * scale for lo in ci_lo]
    ci_hi_d   = [hi * scale for hi in ci_hi]

    # noise band
    ax.axhspan(-0.15, 0.15, xmin=0.04, xmax=0.96,
                color="#f5f5f5", zorder=0, alpha=0.7)
    ax.text(0.96, 0.16, "noise band", transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right", color=COL_NEUTRAL, style="italic")

    for x, md, lo_d, hi_d, c in zip(xs, means_d, ci_lo_d, ci_hi_d, colors):
        ax.bar(x, md, width=bw, color=c, edgecolor="white",
               linewidth=1.2, zorder=3)
        ax.plot([x, x], [lo_d, hi_d], color="black", linewidth=1.3, zorder=4)
        ax.plot([x - 0.018, x + 0.018], [lo_d, lo_d], color="black", linewidth=1.3, zorder=4)
        ax.plot([x - 0.018, x + 0.018], [hi_d, hi_d], color="black", linewidth=1.3, zorder=4)

    # Baseline
    ax.axhline(0, color="black", linewidth=1.0, zorder=2)

    for x, s in zip(xs, substrates):
        w = "bold" if s == "CHO" else "normal"
        ax.text(x, -0.33, s, transform=ax.transAxes, ha="center",
                fontsize=11, fontweight=w,
                color=COL_CHO if s == "CHO" else "black")

    # Annotation showing true magnitude for CHO
    ax.text(0.97, -0.16, r"$\times$0.13 shown",
            transform=ax.transAxes, ha="right", fontsize=6.5,
            color=COL_NEUTRAL, style="italic")

    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_yticklabels(["−30", "−15", "0", "+15", "+30"], fontsize=9)
    ax.set_ylabel("signed SHAP  (CHO × 0.13 for display)", fontsize=9)
    ax.tick_params(axis="y", which="both", length=3, color=COL_NEUTRAL)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(COL_NEUTRAL)
    ax.spines["left"].set_position(("axes", -0.02))

    ax.set_ylim(-0.40, 0.85)
    ax.set_xlim(0, 1)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="black")


# ---------------------------------------------------------------------------
# Panel (b): Real DFT vs xTB HOMO — from 514_dft_vs_xtb_report.csv
# ---------------------------------------------------------------------------
def panel_xtb_dft(ax):
    """Scatter of real DFT (B3LYP-D3BJ/def2-TZVP) vs GFN2-xTB HOMO energies.

    Data: dft_validation/results/514_dft_vs_xtb_report.csv
    15-molecule calibration set; 3 TBAI ion pairs excluded.
    """
    # Read real data
    df = pd.read_csv("dft_validation/results/514_dft_vs_xtb_report.csv")

    # Exclude TBAI ion pairs (documented xTB SCF non-convergence in implicit solvent)
    is_tbai = df["name"].str.contains("TBAI", case=False, na=False)
    df = df[~is_tbai].copy()

    xtb_col = "xtb_homo_eV"
    dft_col = "dft_homo_eV"
    df = df.dropna(subset=[xtb_col, dft_col])

    is_terminal = df["name"].isin([
        "styrene_oxide", "epichlorohydrin", "propylene_oxide",
        "epoxybutane", "isopropyl_glycidyl_ether",
        "allyl_glycidyl_ether", "furfuryl_glycidyl_ether",
        "phenyl_glycidyl_ether",
    ])
    is_cho = df["name"] == "cyclohexene_oxide"

    x = df[xtb_col].values
    y = df[dft_col].values

    # Other molecules (background)
    mask_other = ~(is_terminal | is_cho)
    ax.scatter(x[mask_other], y[mask_other],
               s=45, c=COL_OTHER, edgecolor="white", linewidth=0.6,
               alpha=0.8, zorder=2, label="other (N={})".format(mask_other.sum()))

    # Terminal epoxides
    ax.scatter(x[is_terminal], y[is_terminal],
               s=80, c=COL_TERMINAL, edgecolor="white", linewidth=0.8,
               alpha=0.95, zorder=3, label="terminal epoxide (n={})".format(is_terminal.sum()))

    # CHO (highlighted)
    ax.scatter(x[is_cho], y[is_cho],
               s=140, c=COL_CHO, edgecolor="white", linewidth=1.0,
               marker="D", alpha=1.0, zorder=4,
               label="CHO")

    # Annotate each terminal epoxide
    for xi, yi, name in zip(x[is_terminal], y[is_terminal], df.loc[is_terminal, "name"]):
        short = name.replace("_", " ").title()
        ax.annotate(short,
                    xy=(xi, yi), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5, alpha=0.7)

    # Spearman on clean subset
    rho_all, _ = spearmanr(x, y)
    rho_term, _ = spearmanr(x[is_terminal], y[is_terminal])

    # Panel label
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="black")

    # Stats annotation
    ax.annotate(
        f"All:  Spearman ρ = {rho_all:+.3f}\n"
        f"Terminal: ρ = {rho_term:+.3f}  (n = {is_terminal.sum()})\n"
        f"(TBAI excluded)",
        xy=(-5.4, -4.5), fontsize=9, color=COL_HIGHLIGHT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor=COL_HIGHLIGHT, linewidth=1.2),
        ha="left", va="bottom",
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL_NEUTRAL)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=COL_NEUTRAL, labelsize=9)

    ax.set_xlabel("xTB GFN2 HOMO  (eV)", fontsize=9.5)
    ax.set_ylabel("DFT HOMO  (eV)", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig = plt.figure(figsize=(10, 4.5), dpi=150)
    gs = fig.add_gridspec(1, 2,
                          left=0.08, right=0.98,
                          bottom=0.18, top=0.78,
                          wspace=0.28,
                          width_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    panel_shap(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_xtb_dft(ax_b)

    # Main title
    fig.text(0.5, 0.92,
             "Substrate HOMO Sign Reversal in CO₂ Cycloaddition",
             transform=fig.transFigure,
             fontsize=13, fontweight="bold", ha="center", va="center")

    # Subtitle
    fig.text(0.5, 0.87,
             "Terminal epoxides: positive SHAP  |  CHO: sign reversal (95% CI non-overlap, |t| > 100)",
             transform=fig.transFigure,
             fontsize=9, ha="center", va="center",
             color=COL_NEUTRAL, style="italic")

    # Arrow with label between panels
    arrow_x, arrow_y = 0.50, 0.50
    fig.text(arrow_x, arrow_y + 0.04,
             "rank invariant",
             transform=fig.transFigure, fontsize=8,
             ha="center", va="center",
             color=COL_HIGHLIGHT, fontweight="bold")

    arrow = FancyArrowPatch(
        (arrow_x - 0.04, arrow_y), (arrow_x + 0.04, arrow_y),
        arrowstyle="-|>", mutation_scale=20,
        color=COL_HIGHLIGHT, linewidth=2.0,
        transform=fig.transFigure, zorder=10,
    )
    fig.add_artist(arrow)

    # Footer
    fig.text(0.5, 0.07,
             "n = 2,490  |  5 epoxides  |  LOSO R² = +0.228 ± 0.007  (XGB, PCL-AE 128-D, 3-seed mean)",
             transform=fig.transFigure, fontsize=8.5,
             ha="center", va="center", color=COL_NEUTRAL)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig0_graphical_abstract.png",
               dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig0_graphical_abstract.pdf",
               bbox_inches="tight", facecolor="white")
    print("Saved:", out_dir / "fig0_graphical_abstract.png")
    print("Saved:", out_dir / "fig0_graphical_abstract.pdf")


if __name__ == "__main__":
    main()
