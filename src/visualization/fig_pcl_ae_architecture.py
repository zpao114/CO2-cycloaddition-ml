"""Figure S-A: PCL-AE (Property-supervised Contrastive Learning Autoencoder) architecture.

Box-and-arrow schematic of the PCL-AE-128 model used in this work.
Encoder maps DRFP (2048-D) → latent z (128-D). Two heads read from z:
- Decoder: z → DRFP reconstruction (BCE loss)
- Property head: z → yield prediction (MSE loss, weighted by lambda = {BEST_LAMBDA_PROP})

This is the architecture described in §2.3 of the main text and implemented
in train_pcl_ae_save_latent.py (PCLAE class, lines 86-106).

Output: figures/fig_pcl_ae_architecture.png / .pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import sys
PROJECT_ROOT = Path(os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import BEST_LAMBDA_PROP

# ── Font setup ───────────────────────────────────────────────────────────
avail = [f.name for f in fm.fontManager.ttflist]
cjk = ["simhei", "yahei", "noto sans cjk", "noto sans sc",
       "simsun", "source han", "arial unicode ms"]
cjk_fonts = [f for f in avail if any(k in f.lower() for k in cjk)]
if cjk_fonts:
    plt.rcParams["font.sans-serif"] = cjk_fonts + ["DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Style constants ──────────────────────────────────────────────────────
COL_INPUT    = "#e8eef5"   # light blue-gray for inputs
COL_ENC      = "#5b7fa3"   # mid blue for encoder
COL_LATENT   = "#c0392b"   # red accent for latent (paper's highlight color)
COL_DEC      = "#7d96ae"   # light blue-gray for decoder
COL_PROP     = "#3b8a76"   # teal for property head
COL_ARROW    = "#222222"
COL_TEXT     = "#222222"

LW = 1.6
FS_LABEL = 11
FS_BOX   = 10
FS_DIM   = 9

# ── Helpers ─────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, dim, fc, ec=None, lw=LW, fontsize=FS_BOX):
    if ec is None:
        ec = fc
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    cx, cy = x + w/2, y + h/2
    ax.text(cx, cy + 0.06, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold")
    ax.text(cx, cy - 0.10, dim, ha="center", va="center",
            fontsize=FS_DIM, color="white", style="italic")

def arrow(ax, x1, y1, x2, y2, color=COL_ARROW, lw=1.4, ls="-",
          connectionstyle="arc3,rad=0", mutation_scale=14):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", color=color, lw=lw,
                        linestyle=ls,
                        mutation_scale=mutation_scale,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)

# ── Build figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis("off")

# Title
ax.text(5.5, 5.15, "PCL-AE-128: Property-supervised Contrastive Learning Autoencoder",
        ha="center", va="center", fontsize=14, fontweight="bold", color=COL_TEXT)
ax.text(5.5, 4.80, "Architecture (this work; 80-epoch Adam, lr=1e-3, batch=64, λ=0.1)",
        ha="center", va="center", fontsize=10, style="italic", color="#555555")

# Input layer (left)
box(ax, 0.3, 3.3, 1.6, 0.6, "DRFP input", "2048-D",
    fc=COL_INPUT, ec="#5b7fa3")
# Yield input (left, below)
box(ax, 0.3, 1.3, 1.6, 0.6, "Yield (target)", "scalar",
    fc=COL_INPUT, ec="#3b8a76")

# Encoder stack (3 linear blocks)
box(ax, 2.5, 3.3, 1.3, 0.6, "Linear 2048→512", "BN, ReLU, Drop 0.2", fc=COL_ENC)
box(ax, 4.0, 3.3, 1.3, 0.6, "Linear 512→256", "BN, ReLU, Drop 0.1", fc=COL_ENC)
box(ax, 5.5, 3.3, 1.3, 0.6, "Linear 256→128", "(no nonlinearity)", fc=COL_ENC)

# Latent
box(ax, 7.3, 3.3, 1.3, 0.6, "Latent z", "128-D",
    fc=COL_LATENT, lw=2.0)

# Decoder stack (3 linear blocks, reversed)
box(ax, 5.5, 1.3, 1.3, 0.6, "Linear 128→256", "ReLU", fc=COL_DEC)
box(ax, 4.0, 1.3, 1.3, 0.6, "Linear 256→512", "BN, ReLU", fc=COL_DEC)
box(ax, 2.5, 1.3, 1.3, 0.6, "Linear 512→2048", "Sigmoid", fc=COL_DEC)

# Reconstruction output
box(ax, 0.3, 1.3, 1.6, 0.6, "DRFP recon", "2048-D",
    fc=COL_INPUT, ec="#7d96ae")

# Property head (right column)
box(ax, 9.0, 3.3, 1.5, 0.6, "Linear 128→64", "ReLU, Drop 0.1", fc=COL_PROP)
box(ax, 9.0, 2.4, 1.5, 0.6, "Linear 64→32", "ReLU", fc=COL_PROP)
box(ax, 9.0, 1.5, 1.5, 0.6, "Linear 32→1", "(scalar)", fc=COL_PROP)

# Property output
box(ax, 9.0, 0.5, 1.5, 0.6, "Yield prediction", "scalar",
    fc=COL_INPUT, ec="#3b8a76")

# ── Arrows: input → encoder ──
arrow(ax, 1.9, 3.6, 2.5, 3.6)
arrow(ax, 1.9, 1.6, 2.5, 1.6)  # yield also fed into loss (not encoder)

# Encoder stack
arrow(ax, 3.8, 3.6, 4.0, 3.6)
arrow(ax, 5.3, 3.6, 5.5, 3.6)

# Encoder → latent
arrow(ax, 6.8, 3.6, 7.3, 3.6, lw=2.0)

# Latent → decoder (down)
arrow(ax, 7.95, 3.3, 6.8, 1.9, connectionstyle="angle3,angleA=0,angleB=90",
      color=COL_LATENT, lw=1.6)

# Latent → property head (up-right)
arrow(ax, 8.6, 3.6, 9.0, 3.6, color=COL_LATENT, lw=1.6)

# Decoder stack (right to left)
arrow(ax, 5.5, 1.6, 4.3, 1.6)
arrow(ax, 4.0, 1.6, 2.8, 1.6)
# Decoder → recon
arrow(ax, 2.5, 1.6, 1.9, 1.6)

# Property head stack (top to bottom)
arrow(ax, 9.75, 3.3, 9.75, 3.0)
arrow(ax, 9.75, 2.4, 9.75, 2.1)
arrow(ax, 9.75, 1.5, 9.75, 1.1)
# Property → yield prediction
# (already at right place visually; arrow implicit)

# ── Loss annotations ──
# Loss 1: recon loss (BCE) - below decoder path
ax.text(3.9, 0.7, "L_recon = BCE(recon, DRFP)",
        ha="center", va="center", fontsize=10,
        color=COL_DEC, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL_DEC, lw=1.2))

# Loss 2: property loss (MSE) - below property head
ax.text(9.75, 0.7, "L_prop = MSE(pred, yield)",
        ha="center", va="center", fontsize=10,
        color=COL_PROP, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL_PROP, lw=1.2))

# Total loss
ax.text(5.5, 0.25,
        r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{recon}} + \lambda \cdot \mathcal{L}_{\mathrm{prop}},\quad \lambda = " + str(BEST_LAMBDA_PROP) + r"$",
        ha="center", va="center", fontsize=12,
        color=COL_TEXT, fontweight="bold")

arrow(ax, 5.5, 0.45, 4.5, 0.55, color="#555555", lw=0.8, ls=":")
arrow(ax, 5.5, 0.45, 8.5, 0.55, color="#555555", lw=0.8, ls=":")

# ── Legend (top right corner) ──
legend_elements = [
    Line2D([0], [0], color=COL_INPUT, marker="s", linestyle="",
           markerfacecolor=COL_INPUT, markeredgecolor="#5b7fa3",
           markersize=10, label="Inputs / outputs"),
    Line2D([0], [0], color=COL_ENC, marker="s", linestyle="",
           markerfacecolor=COL_ENC, markersize=10, label="Encoder"),
    Line2D([0], [0], color=COL_LATENT, marker="s", linestyle="",
           markerfacecolor=COL_LATENT, markersize=10, label="Latent z"),
    Line2D([0], [0], color=COL_DEC, marker="s", linestyle="",
           markerfacecolor=COL_DEC, markersize=10, label="Decoder"),
    Line2D([0], [0], color=COL_PROP, marker="s", linestyle="",
           markerfacecolor=COL_PROP, markersize=10, label="Property head"),
]
ax.legend(handles=legend_elements, loc="upper left",
          bbox_to_anchor=(0.0, 1.05),
          frameon=True, framealpha=0.95,
          fontsize=9, ncol=2)

# ── Save ────────────────────────────────────────────────────────────────
out_dir = Path(r"D:\machine-learning\CO2-cycloaddition\figures")
out_dir.mkdir(exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(out_dir / f"fig_pcl_ae_architecture.{ext}",
                dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_dir / 'fig_pcl_ae_architecture.png'}")
print(f"Saved: {out_dir / 'fig_pcl_ae_architecture.pdf'}")
