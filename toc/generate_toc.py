#!/usr/bin/env python3
"""TOC graphic for the RSEP manuscript (v12, minimal three-stage design).

One reading line, left to right:

  DATA      1,033 CO2-cycloaddition reactions (Reaxys, 5 epoxides) enter
            the protocol stage once.
  PROTOCOL  that single dataset is evaluated twice: Pretrain-Then-Freeze
            (leaky, red, dashed) and Per-Fold Retrain (honest, teal); the
            bracket between the two outcomes is the quantity read off next.
  RESULT    Delta_protocol = +0.40 for PCL-AE, with the five per-substrate
            values as a Cleveland dot plot: every substrate lies on the
            positive side of zero and cyclohexene oxide (CHO) sits far out.

Colour carries exactly one contrast: red = leaked / inflated evaluation,
teal = honest evaluation. Everything else (molecules, axes, captions) is
neutral grey or near-black, so the two accents stay legible when the
figure is reduced to contents-page width.

Dropped relative to earlier drafts (database cylinder, ladder path, per-bar
value labels, molecule badge, formula block): at 8.25 x 4.45 cm a TOC has
room for one argument, not a figure legend.

ACS TOC canvas: 3.25 x 1.75 in (8.25 x 4.45 cm) at 600 dpi, pure white,
6 pt minimum lettering (heteroatom letters in the line-art molecules are
exempt), 1 data unit = 1 pt, PNG + SVG.
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

# ----------------------------------------------------------------------------
# Canvas and palette
# ----------------------------------------------------------------------------

DPI = 600
W_IN, H_IN = 3.25, 1.75
PT_W, PT_H = W_IN * 72.0, H_IN * 72.0

C_BG = "#FFFFFF"
C_TXT = "#0F172A"
C_SUB = "#64748B"
C_RULE = "#E9EEF5"

C_LEAK = "#B3261E"
C_HONEST = "#0F766E"
C_DOT = "#94A3B8"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.default"] = "regular"

fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_subplot(111)
ax.set_xlim(0, PT_W)
ax.set_ylim(0, PT_H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0.0, 0.0, 1.0, 1.0])

fig.canvas.draw()
RENDERER = fig.canvas.get_renderer()


def measure(text, size, weight="normal"):
    t = ax.text(0, 0, text, fontsize=size, weight=weight)
    bb = t.get_window_extent(renderer=RENDERER)
    t.remove()
    return bb.width * 72.0 / DPI, bb.height * 72.0 / DPI


def require_fits(text, size, weight, avail, where):
    w, _ = measure(text, size, weight)
    if w > avail:
        raise SystemExit(f"LAYOUT ERROR: '{text}' needs {w:.2f} pt, "
                         f"{where} allows {avail:.2f} pt")
    return w


EXEMPT = set()


def arrow(p1, p2, color, lw=0.8, ms=4.6, zorder=3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0,
                                 zorder=zorder))


def rounded(x, y, w, h, ec, ls="solid", lw=0.9, radius=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=3))


# ----------------------------------------------------------------------------
# Molecular line art (RDKit 2D coordinates, normalised to a unit box)
# ----------------------------------------------------------------------------


def mol_geometry(smiles, rot_deg=90.0):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
           for i in range(mol.GetNumAtoms())}
    ca, sa = math.cos(math.radians(-rot_deg)), math.sin(math.radians(-rot_deg))
    pts = {i: (x * ca - y * sa, -(x * sa + y * ca)) for i, (x, y) in pts.items()}
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    ox, oy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    pts = {i: ((x - ox) / span, (y - oy) / span) for i, (x, y) in pts.items()}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
             for b in mol.GetBonds()]
    hetero = {i: a.GetSymbol() for i, a in enumerate(mol.GetAtoms())
              if a.GetSymbol() != "C"}
    return pts, bonds, hetero


GEOM_EPOX = mol_geometry("C1CO1")
GEOM_CARB = mol_geometry("O=C1OCCO1")


def draw_skeleton(geom, cx, cy, height, color, lw=0.75, o_size=5.0):
    pts = {i: (cx + v[0] * height, cy + v[1] * height)
           for i, v in geom[0].items()}
    r = o_size * 0.40 + lw * 0.5
    for a, b, bt in geom[1]:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        if a in geom[2]:
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            x1, y1 = x1 + dx / d * r, y1 + dy / d * r
        if b in geom[2]:
            dx, dy = x1 - x2, y1 - y2
            d = math.hypot(dx, dy)
            x2, y2 = x2 + dx / d * r, y2 + dy / d * r
        if bt == Chem.BondType.DOUBLE:
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            pxn, pyn = -dy / d, dx / d
            for s in (-1.0, 1.0):
                ax.plot([x1 + pxn * 0.9 * s, x2 + pxn * 0.9 * s],
                        [y1 + pyn * 0.9 * s, y2 + pyn * 0.9 * s],
                        color=color, lw=lw * 0.9, solid_capstyle="butt",
                        zorder=4)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                    solid_capstyle="round", zorder=4)
    for i, sym in geom[2].items():
        x, y = pts[i]
        EXEMPT.add(ax.text(x, y, sym, ha="center", va="center", fontsize=o_size,
                           color=color, weight="bold", zorder=5))


def draw_co2(cx, cy, w, color, lw=0.7, o_size=5.0):
    xl, xr = cx - w / 2, cx + w / 2
    r = o_size * 0.40 + lw * 0.5
    for dy in (-0.8, 0.8):
        ax.plot([xl + r, cx - r], [cy + dy, cy + dy], color=color,
                lw=lw * 0.9, solid_capstyle="butt", zorder=4)
        ax.plot([cx + r, xr - r], [cy + dy, cy + dy], color=color,
                lw=lw * 0.9, solid_capstyle="butt", zorder=4)
    for xo in (xl, xr):
        EXEMPT.add(ax.text(xo, cy, "O", ha="center", va="center",
                           fontsize=o_size, color=color, weight="bold",
                           zorder=5))
    EXEMPT.add(ax.text(cx, cy, "C", ha="center", va="center", fontsize=o_size,
                       color=color, weight="bold", zorder=5))


# ----------------------------------------------------------------------------
# Data (read from the authoritative multiseed summary)
# ----------------------------------------------------------------------------

with open(os.path.join(REPO, "results_pcl_ae_fair_comparison_multiseed",
                       "multiseed_summary.json"), encoding="utf-8") as fh:
    SUMMARY = json.load(fh)
PCL = SUMMARY["delta_protocol"]["PCL-AE"]
MEAN = float(PCL["mean"])
ABBR = {"Cyclohexene oxide": "CHO", "Epichlorohydrin": "ECH",
        "Isopropyl glycidyl ether": "IGE", "Propylene oxide": "PO",
        "Styrene oxide": "SO"}
DOTS = sorted(((ABBR[k], float(v)) for k, v in
               PCL["substrate_level"]["per_substrate"].items()),
              key=lambda t: -t[1])  # CHO first, PO last

# ----------------------------------------------------------------------------
# Stage 1 - the data (x 4 .. 66), centred on the fork height
# ----------------------------------------------------------------------------

CHEM_Y = 74.0

draw_co2(13.0, CHEM_Y, 11.0, C_TXT, o_size=5.0)
ax.text(24.5, CHEM_Y, "+", ha="center", va="center", fontsize=7.0,
        weight="bold", color=C_TXT, zorder=3)
draw_skeleton(GEOM_EPOX, 34.5, CHEM_Y, 11.0, C_TXT, o_size=4.6)
arrow((41.5, CHEM_Y), (46.0, CHEM_Y), C_TXT)
draw_skeleton(GEOM_CARB, 54.0, CHEM_Y, 13.0, C_TXT, o_size=4.8)

require_fits("1,033 reactions", 7.0, "bold", 60.0, "stage 1 count")
require_fits("Reaxys \u00b7 5 epoxides", 6.0, "normal", 60.0, "stage 1 source")
require_fits("leave-one-substrate-out", 6.0, "normal", 68.0, "stage 1 design")
ax.text(33.5, 55.0, "1,033 reactions", ha="center", va="center",
        fontsize=7.0, weight="bold", color=C_TXT, zorder=3)
ax.text(33.5, 43.0, "Reaxys \u00b7 5 epoxides", ha="center", va="center",
        fontsize=6.0, color=C_SUB, zorder=3)
ax.text(33.5, 31.0, "leave-one-substrate-out", ha="center", va="center",
        fontsize=6.0, color=C_SUB, zorder=3)

# ----------------------------------------------------------------------------
# Stage 2 - one dataset, two protocols (x 70 .. 150)
# ----------------------------------------------------------------------------

SPINE_X = 70.0
Y_LEAK, Y_HONEST = 92.0, 52.0
BOX_X0, BOX_W, BOX_H = 93.0, 46.0, 18.0
BOX_CX = BOX_X0 + BOX_W / 2

# The single arrow from stage 1 lands on the middle of the fork, which then
# splits: one dataset, the same folds and seeds, two encoder workflows.
arrow((60.0, CHEM_Y), (SPINE_X, CHEM_Y), C_TXT, lw=0.9)
ax.plot([SPINE_X, SPINE_X], [Y_HONEST, Y_LEAK], color=C_SUB, lw=0.8,
        solid_capstyle="round", zorder=2)

require_fits("Same data, two protocols", 6.0, "bold", 96.0, "stage 2 header")
ax.text(116.0, 112.0, "Same data, two protocols", ha="center", va="top",
        fontsize=6.0, weight="bold", color=C_TXT, zorder=3)

arrow((SPINE_X, Y_LEAK), (BOX_X0 - 1.5, Y_LEAK), C_LEAK, lw=0.9)
rounded(BOX_X0, Y_LEAK - BOX_H / 2, BOX_W, BOX_H, C_LEAK, ls=(0, (2.2, 1.3)))
ax.text(BOX_CX, Y_LEAK, "Pretrain-\nThen-Freeze", ha="center", va="center",
        fontsize=6.0, weight="bold", color=C_LEAK, linespacing=1.25, zorder=4)

arrow((SPINE_X, Y_HONEST), (BOX_X0 - 1.5, Y_HONEST), C_HONEST, lw=0.9)
rounded(BOX_X0, Y_HONEST - BOX_H / 2, BOX_W, BOX_H, C_HONEST)
ax.text(BOX_CX, Y_HONEST, "Per-Fold\nRetrain", ha="center", va="center",
        fontsize=6.0, weight="bold", color=C_HONEST, linespacing=1.25, zorder=4)

# The distance between the two outcomes is the quantity the last stage reads.
BR_X = 146.5
ax.plot([BR_X, BR_X], [Y_HONEST, Y_LEAK], color=C_LEAK, lw=0.8, zorder=2)
for yy in (Y_HONEST, Y_LEAK):
    ax.plot([BR_X - 2.2, BR_X], [yy, yy], color=C_LEAK, lw=0.8, zorder=2)
ax.text(BR_X - 3.2, (Y_LEAK + Y_HONEST) / 2, r"$\Delta$", ha="right",
        va="center", fontsize=7.0, color=C_LEAK, zorder=3)

# The subtraction formula (the v11 idiom, kept): what the bracket IS.
# mathtext has no \color, so it is drawn as four coloured pieces.
_FORMULA = [
    (r"$\Delta_{\mathrm{protocol}} =$", C_TXT),
    (r"$R^{2}_{\mathrm{pretrain}}$", C_LEAK),
    (r"$-$", C_TXT),
    (r"$R^{2}_{\mathrm{retrain}}$", C_HONEST),
]
_F_GAP = 1.0
_F_W = sum(measure(t, 6.0)[0] for t, _ in _FORMULA) + _F_GAP * (len(_FORMULA) - 1)
_F_X = 116.0 - _F_W / 2.0
for _t, _c in _FORMULA:
    ax.text(_F_X, 26.0, _t, ha="left", va="center", fontsize=6.0,
            color=_c, zorder=3)
    _F_X += measure(_t, 6.0)[0] + _F_GAP

# ----------------------------------------------------------------------------
# Stage 3 - the result (x 152 .. 234)
# ----------------------------------------------------------------------------

S3_CX = 193.0
require_fits(r"$\Delta_{\mathrm{protocol}}$ (LOSO)", 6.0, "normal",
             78.0, "stage 3 header")
ax.text(S3_CX, 112.0, r"$\Delta_{\mathrm{protocol}}$ (LOSO)",
        ha="center", va="top", fontsize=6.0, color=C_SUB, zorder=3)

ax.text(S3_CX, 95.0, f"+{MEAN:.2f}", ha="center", va="center",
        fontsize=16.0, weight="bold", color=C_LEAK, zorder=3)
ax.text(S3_CX, 82.0, "PCL-AE", ha="center", va="center", fontsize=6.0,
        color=C_LEAK, zorder=3)

X0, X1 = 174.0, 214.0
AX_MAX = 1.10
AX_Y = 26.0
ROW_Y = [70.0 - i * 9.0 for i in range(5)]
VAL_X = 231.5
LAB_X = X0 - 2.2

ax.plot([X0, X1], [AX_Y, AX_Y], color=C_SUB, lw=0.6, zorder=2)
ax.plot([X0, X0], [AX_Y, ROW_Y[0] + 3.0], color=C_SUB, lw=0.6, zorder=2)
ax.text(X0, AX_Y - 4.0, "0", ha="center", va="top", fontsize=6.0,
        color=C_SUB, zorder=3)
ax.text(X1, AX_Y - 4.0, f"{AX_MAX:.1f}", ha="center", va="top", fontsize=6.0,
        color=C_SUB, zorder=3)

for (name, val), y in zip(DOTS, ROW_Y):
    xv = X0 + val / AX_MAX * (X1 - X0)
    hot = name == "CHO"
    col = C_LEAK if hot else C_DOT
    ax.plot([X0, X1], [y, y], color=C_RULE, lw=0.4, zorder=1)
    ax.plot([X0, xv], [y, y], color=col, lw=0.8, zorder=3)
    ax.plot([xv], [y], marker="o", markersize=2.8, color=col, zorder=4)
    ax.text(LAB_X, y, name, ha="right", va="center", fontsize=6.0,
            weight="bold" if hot else "normal", color=col, zorder=3)
    ax.text(VAL_X, y, f"{val:.2f}", ha="right", va="center", fontsize=6.0,
            weight="bold" if hot else "normal", color=col, zorder=3)

require_fits("all 5 substrates positive", 6.0, "normal", 78.0, "stage 3 note")
ax.text(S3_CX, 9.0, "all 5 substrates positive", ha="center", va="center",
        fontsize=6.0, color=C_SUB, zorder=3)

# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

OUT_PNG = os.path.join(HERE, "toc_graphic.png")
OUT_SVG = os.path.join(HERE, "toc_graphic.svg")

fig.savefig(OUT_PNG, dpi=DPI, facecolor=C_BG, edgecolor="none")
fig.savefig(OUT_SVG, format="svg", facecolor=C_BG, edgecolor="none")

Image.open(OUT_PNG).convert("RGB").save(OUT_PNG, dpi=(DPI, DPI))

SYNOPSIS = f"""TOC SYNOPSIS

OPTION 1 - Standard
The same 1,033 CO2 cycloaddition reactions, folds, seeds and regressor are
evaluated under two encoder workflows: pretrain-then-freeze and per-fold
retraining. The paired difference isolates evaluation inflation from genuine
chemical generalisation: the mainstream workflow overstates leave-one-
substrate-out R2 of a property-supervised autoencoder by {MEAN:.2f} on average
(cyclohexene oxide, {DOTS[0][1]:.2f}), while an unsupervised autoencoder is
unaffected. Four minimum reporting standards are proposed.

OPTION 2 - Ultra-short
Pretraining an encoder on the full dataset and freezing it across folds
inflates leave-one-substrate-out R2 by {MEAN:.2f} on average
(cyclohexene oxide, {DOTS[0][1]:.2f}). A paired protocol comparison separates
this evaluation artifact from genuine chemical generalisation and yields four
minimum reporting standards for reaction-yield models.
"""
with open(os.path.join(HERE, "toc_synopsis.txt"), "w", encoding="utf-8") as fh:
    fh.write(SYNOPSIS)

# ----------------------------------------------------------------------------
# Collision and font-floor audit
# ----------------------------------------------------------------------------


def audit():
    fig.canvas.draw()
    inv = 72.0 / DPI
    boxes = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=RENDERER)
        boxes.append((t.get_text().replace("\n", " ")[:24], bb.x0 * inv,
                      bb.y0 * inv, bb.x1 * inv, bb.y1 * inv, t))
    problems = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            dx = min(a[3], b[3]) - max(a[1], b[1])
            dy = min(a[4], b[4]) - max(a[2], b[2])
            if dx > 0.3 and dy > 0.3:
                tag = "[line art]" if (a[5] in EXEMPT or b[5] in EXEMPT) else ""
                problems.append(f"TEXT x TEXT '{a[0]}' <-> '{b[0]}' "
                                f"{dx * dy:.2f} pt2 {tag}")
        if a[1] < 0 or a[3] > PT_W or a[2] < 0 or a[4] > PT_H:
            problems.append(f"OUT OF CANVAS '{a[0]}'")
    exempt = 0
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        if t in EXEMPT:
            exempt += 1
            continue
        if t.get_fontsize() < 6.0:
            problems.append(f"SMALL FONT '{t.get_text()[:20]}' "
                            f"{t.get_fontsize():.1f} pt")
    if problems:
        print(f"[WARN] {len(problems)} layout problem(s):")
        for p in problems:
            print("   -", p)
    else:
        print("[OK] audit passed: no text overlap, all text on canvas, "
              f"{exempt} line-art label(s) exempt from the 6 pt floor")
    return problems


problems = audit()

print(f"[OK] toc_graphic -> {OUT_PNG}, {OUT_SVG}")
print(f"     canvas {W_IN * 2.54:.2f} x {H_IN * 2.54:.2f} cm @ {DPI} dpi")
print(f"     headline +{MEAN:.2f}, per-substrate "
      + ", ".join(f"{n} {v:.2f}" for n, v in DOTS))
