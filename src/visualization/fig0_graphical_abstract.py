"""Graphical Abstract for paper_final.

Four-step workflow flow, replacing the prior SHAP-DRFP two-panel layout.
Per the Top-7 fix (GA/Figure-4 redundancy), the graphical abstract now
presents the four-stage diagnostic protocol rather than repeating the
SHAP-direction bar chart (Figure 4).

Pipeline:
  Stage 1: Data governance (n=2,490; 5 substrates; 5 catalyst classes)
  Stage 2: Four-tier evaluation ladder (GroupKFold → LOMO → LOSO → External)
  Stage 3: Three evidence lines converging on sub_homo_eV (SHAP CI / KL 5.5× / permutation v2+v3)
  Stage 4: Actionable targets (CHO × organic_base n=9; MAE > R^2 for deployment)

Outputs:
  fig0_graphical_abstract_en.{pdf,png,svg}   (English caption inside figure)
  fig0_graphical_abstract_zh.{pdf,png,svg}   (Chinese caption inside figure)
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# Ensure CJK glyph fallback for the Chinese title (Windows machines).
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Color palette (color-blind aware)
# ---------------------------------------------------------------------------
COL_TERMINAL = "#1f77b4"    # blue: stage headers / terminal epoxides
COL_CHO = "#d62728"         # red: CHO / failure mode
COL_NEUTRAL = "#7f7f7f"     # grey: secondary text
COL_XTB = "#2ca02c"         # green: xTB / evidence confirmed
COL_DFT = "#9467bd"         # purple: B3LYP-D3BJ
COL_HIGHLIGHT = "#ff7f0e"   # orange: actionable call-out
COL_BOX_EDGE = "#404040"
COL_BOX_FILL_TINT = "#fafafa"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _rounded_box(ax, x, y, w, h, face, edge, lw=1.4, radius=0.025, zorder=2):
    """Rounded rectangle using a single FancyBboxPatch (axes-fraction coords)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _stage_box(ax, x, y, w, h, title, body_lines, accent_color, icon_text=None):
    """One workflow stage: title bar + body lines + icon glyph."""
    # Outer box
    _rounded_box(ax, x, y, w, h, face=COL_BOX_FILL_TINT,
                 edge=COL_BOX_EDGE, lw=1.3, radius=0.020, zorder=2)
    # Title bar (top 28%)
    bar_h = h * 0.30
    title_box = FancyBboxPatch(
        (x, y + h - bar_h), w, bar_h,
        boxstyle="round,pad=0.002,rounding_size=0.018",
        facecolor=accent_color, edgecolor=accent_color,
        linewidth=0.0, transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(title_box)
    # Stage number + title text
    n_text = icon_text if icon_text else ""
    ax.text(x + w * 0.04, y + h - bar_h * 0.5, n_text,
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="center", color="white")
    ax.text(x + w * 0.18, y + h - bar_h * 0.5, title,
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="center", color="white")

    # Body lines
    n_body = len(body_lines)
    body_top = y + h - bar_h - 0.010
    body_bot = y + 0.012
    row_h = (body_top - body_bot) / max(n_body, 1)
    for i, line in enumerate(body_lines):
        cy = body_top - row_h * (i + 0.5)
        ax.text(x + w * 0.05, cy, line,
                transform=ax.transAxes, fontsize=8.0,
                ha="left", va="center", color="black")


def _connector(ax, x0, x1, y, color=COL_HIGHLIGHT, lw=2.0):
    """Arrow connector between two stages."""
    arrow = FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=18,
        color=color, linewidth=lw,
        transform=ax.transAxes, zorder=6,
    )
    ax.add_patch(arrow)


def main():
    fig = plt.figure(figsize=(13.5, 5.4), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -------------------------------------------------------------------------
    # Figure-level title block (English by default; Chinese variant uses
    # dedicated runner below).
    # -------------------------------------------------------------------------
    fig.text(0.5, 0.945,
             "Mechanism-aware diagnostics reveal cross-substrate "
             "extrapolation failure in CO\u2082 cycloaddition yield prediction",
             ha="center", va="bottom",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.905,
             "2,490 Reaxys reactions \u00b7 five epoxides \u00b7 "
             "five catalyst system classes \u00b7 "
             "four-tier evaluation \u00b7 three evidence lines",
             ha="center", va="bottom",
             fontsize=9.5, color=COL_NEUTRAL, style="italic")

    # -------------------------------------------------------------------------
    # Four stages laid out left-to-right at vertical center
    # -------------------------------------------------------------------------
    margin_x = 0.020
    arrow_x = 0.030
    stage_w = 0.215
    stage_h = 0.55
    cy = 0.52
    by = cy - stage_h / 2

    x1 = margin_x
    x2 = x1 + stage_w + arrow_x
    x3 = x2 + stage_w + arrow_x
    x4 = x3 + stage_w + arrow_x

    # Stage 1: Data
    _stage_box(ax, x1, by, stage_w, stage_h,
               title="Data governance",
               body_lines=[
                   "n = 2,490 reactions",
                   "5 epoxides (PO, ECH, SO,",
                   "    CHO, IGE)",
                   "5 catalyst classes",
                   "1970\u20132024 (Reaxys)",
                   "yield \u2208 [0, 100]%",
                   "81.9% > 70% (right-skewed)",
               ],
               accent_color=COL_TERMINAL,
               icon_text="1")

    # Stage 2: Protocol ladder
    _stage_box(ax, x2, by, stage_w, stage_h,
               title="Four-tier ladder",
               body_lines=[
                   "(a) 5-fold GroupKFold",
                   "    groups = catalyst_type",
                   "(b) LOMO",
                   "    groups = mechanism",
                   "(c) LOSO",
                   "    groups = substrate",
                   "(d) External holdout",
               ],
               accent_color=COL_XTB,
               icon_text="2")

    # Stage 3: Three evidence lines
    _stage_box(ax, x3, by, stage_w, stage_h,
               title="Three converging lines",
               body_lines=[
                   "SHAP CI: CHO [\u221225.61,",
                   "    \u221224.68] vs PO",
                   "    [+3.60, +3.73] (gap 28.28)",
                   "KL: CHO\u2013terminal 14.35",
                   "    vs LOMO 2.59 (\u00d75.5)",
                   "Permutation v2/v3:",
                   "    genuine on terminals,",
                   "    ambiguous on CHO (p=0.78)",
               ],
               accent_color=COL_DFT,
               icon_text="3")

    # Stage 4: Deployable targets
    _stage_box(ax, x4, by, stage_w, stage_h,
               title="Deployable action",
               body_lines=[
                   "CHO \u00d7 organic_base",
                   "    n = 9, \u03bc = 0.588",
                   "    single-experiment target",
                   "Deployment MAE > R\u00b2",
                   "    4-seed MAE 0.134 \u00b1 0.007",
                   "    95% CI [\u22120.275, +0.537]",
                   "  cannot reject R\u00b2 = 0",
               ],
               accent_color=COL_CHO,
               icon_text="4")

    # -------------------------------------------------------------------------
    # Arrows between stages
    # -------------------------------------------------------------------------
    _connector(ax, x1 + stage_w + 0.004, x2 - 0.004, cy)
    _connector(ax, x2 + stage_w + 0.004, x3 - 0.004, cy)
    _connector(ax, x3 + stage_w + 0.004, x4 - 0.004, cy)

    # -------------------------------------------------------------------------
    # Bottom panels: key headline numbers (mono-row strip)
    # -------------------------------------------------------------------------
    band_y = 0.07
    band_h = 0.16
    band_x = margin_x
    band_w = (1 - 2 * margin_x)
    _rounded_box(ax, band_x, band_y, band_w, band_h,
                 face="#f5f5f5", edge=COL_BOX_EDGE, lw=0.8, radius=0.012, zorder=2)
    headline_cols = [
        ("LOSO (cached)",            "R\u00b2 = +0.228 \u00b1 0.008", COL_TERMINAL),
        ("LOSO (honest per-fold)",   "R\u00b2 = \u22120.816 \u00b1 0.242", COL_CHO),
        ("Leakage gap",              "\u0394R\u00b2 = +1.067 \u00b1 0.242", COL_HIGHLIGHT),
        ("External holdout",         "R\u00b2 = +0.131 \u00b1 0.128", COL_NEUTRAL),
    ]
    cell_w = band_w / len(headline_cols)
    for i, (label, value, color) in enumerate(headline_cols):
        cx = band_x + cell_w * (i + 0.5)
        ax.text(cx, band_y + band_h * 0.65, value,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(cx, band_y + band_h * 0.30, label,
                ha="center", va="center",
                fontsize=8.8, color=COL_NEUTRAL, transform=ax.transAxes)

    # -------------------------------------------------------------------------
    # Footer provenance line
    # -------------------------------------------------------------------------
    fig.text(0.5, 0.035,
             "Data: cleaned.csv \u00b7 bootstrap_substrate_shap_ci.csv \u00b7 "
             "loso_kl_divergence.csv \u00b7 permuted_feature_ablation_v2/v3.csv \u00b7 "
             "external_holdout/multiseed_v2_summary.csv  \u00b7  "
             "n = 2,490  \u00b7  CC BY 4.0",
             ha="center", va="bottom",
             fontsize=7.8, color=COL_NEUTRAL, style="italic")

    return fig


def save_figure(fig, suffix):
    """Save PNG, PDF, SVG with the given suffix (e.g., 'en', 'zh')."""
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig0_graphical_abstract_{suffix}"
    fig.savefig(out_dir / f"{stem}.png",  dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf",          bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg",          bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_dir / stem}.png  .pdf  .svg")


def render_zh_title(fig):
    """Overlay Chinese title for the ZH variant.

    Only the figure-level title and footer are translated; the workflow
    body remains in English to ensure a unified bilingual visual across
    both preprints and to avoid duplicate figure-text workloads. This
    satisfies the 'GA/Figure-4 redundancy' fix (Top 7 #1) for both the
    EN and ZH versions because the figure content is the four-stage
    workflow, not the SHAP direction bar chart.
    """
    cover = mpatches.Rectangle((0, 0.86), 1, 0.135,
                               transform=fig.transFigure,
                               facecolor="white", edgecolor="none", zorder=50)
    fig.add_artist(cover)
    fig.text(0.5, 0.945,
             "\u57fa\u4e8e\u673a\u5236\u611f\u77e5\u8bca\u65ad\u7684CO\u2082\u73af\u52a0\u6210\u4ea7\u7387\u9884\u6d4b"
             "\u63ed\u793a\u4e86\u4e0e\u5e95\u7269\u2013\u673a\u5236\u6b63\u4ea4\u6027\u4e00\u81f4\u7684\u5916\u63a8\u5931\u6548",
             ha="center", va="bottom",
             fontsize=12, fontweight="bold", zorder=51)
    fig.text(0.5, 0.905,
             "2,490 \u4e2aReaxys\u53cd\u5e94 \u00b7 \u4e94\u79cd\u73af\u6c27\u5316\u7269 \u00b7 "
             "\u4e94\u7c7b\u50ac\u5316\u5242 \u00b7 "
             "\u56db\u5c42\u7ea7\u8bc4\u4f30 \u00b7 \u4e09\u6761\u8bc1\u636e\u7ebf",
             ha="center", va="bottom",
             fontsize=9.0, color=COL_NEUTRAL, style="italic", zorder=51)

    # Footer provenance line (Chinese)
    cover_ft = mpatches.Rectangle((0, 0.0), 1, 0.06,
                                  transform=fig.transFigure,
                                  facecolor="white", edgecolor="none", zorder=50)
    fig.add_artist(cover_ft)
    fig.text(0.5, 0.030,
             "\u6570\u636e: cleaned.csv \u00b7 bootstrap_substrate_shap_ci.csv \u00b7 "
             "loso_kl_divergence.csv \u00b7 permuted_feature_ablation_v2/v3.csv \u00b7 "
             "external_holdout/multiseed_v2_summary.csv  \u00b7  "
             "n = 2,490",
             ha="center", va="bottom",
             fontsize=7.8, color=COL_NEUTRAL, style="italic", zorder=51)


def fig_to_ax(fig):
    """Helper: return a hidden axes covering the whole figure, used for overlay boxes."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


if __name__ == "__main__":
    # English variant
    fig_en = main()
    save_figure(fig_en, "en")
    plt.close(fig_en)

    # Chinese variant (re-render with overlay)
    fig_zh = main()
    render_zh_title(fig_zh)
    save_figure(fig_zh, "zh")
    plt.close(fig_zh)

    print("Fig0 (GA) flow chart regenerated for both EN and ZH.")
