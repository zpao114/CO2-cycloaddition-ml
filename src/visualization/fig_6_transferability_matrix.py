# -*- coding: utf-8 -*-
"""
figures/fig_6_transferability_matrix.py
========================================
Fig. 6 — 5×5 底物 × 机制矩阵 (n 覆盖率 + 平均产率)。

主信息：
  (a) 覆盖率矩阵：n < 10 的 5 个单元（IGE×{LAC, BIF, BAS, UNK}, CHO×BAS）用红色边框
  (b) 产率矩阵：CHO 在 5 机制单元均 < 68% (其他 4 底物均 > 75%)，t 检验显著

输入：results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv
输出：fig6_transferability_matrix.png / .pdf
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

ROOT = r"D:\machine-learning\CO2-cycloaddition"
os.chdir(ROOT)

# ── Font ────────────────────────────────────────────────────────────────────
avail = [f.name for f in fm.fontManager.ttflist]
cjk = ["simhei", "wenquanyi", "yahei", "noto sans cjk", "noto sans sc",
       "microsoft yahei", "source han", "arial unicode ms"]
cjk_fonts = [f for f in avail if any(k in f.lower() for k in cjk)]
if cjk_fonts:
    plt.rcParams["font.sans-serif"] = [cjk_fonts[0]] + plt.rcParams["font.sans-serif"]
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv('results/results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv')

SUBSTRATE_ORDER = [
    "Propylene oxide",
    "Styrene oxide",
    "Epichlorohydrin",
    "Cyclohexene oxide",
    "Isopropyl glycidyl ether",
]
SUBSTRATE_LABEL = {
    "Propylene oxide":           "PO",
    "Styrene oxide":             "SO",
    "Epichlorohydrin":           "ECH",
    "Cyclohexene oxide":         "CHO\n(环内)",
    "Isopropyl glycidyl ether":  "IGE",
}

MECH_ORDER = ["ionic_liquid", "metal_halide", "mixed_system", "organic_base", "unknown"]
MECH_LABEL = {
    "ionic_liquid":  "IL\n(NUC)",
    "metal_halide":  "MH\n(LAC)",
    "mixed_system":  "Mixed\n(BIF)",
    "organic_base":  "Base\n(BAS)",
    "unknown":       "Unk.\n(OTH)",
}

pivot_n     = df.pivot_table(index="reactant", columns="catalyst_family",
                             values="n", aggfunc="first"
                             ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)
pivot_yield = df.pivot_table(index="reactant", columns="catalyst_family",
                             values="yield_mean", aggfunc="first"
                             ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0),
                                gridspec_kw={"width_ratios": [1, 1]})
fig.patch.set_facecolor("white")

# ---------- (a) 覆盖率热图 ----------
n_arr = pivot_n.values.astype(float)
n_disp = np.where(np.isnan(n_arr), -1, n_arr)   # NaN -> -1 用于区分
im1 = ax1.imshow(np.where(np.isnan(n_arr), 0, n_arr), cmap="YlGnBu",
                 aspect="auto", vmin=0, vmax=600)
ax1.set_xticks(range(len(MECH_ORDER)))
ax1.set_xticklabels([MECH_LABEL[m] for m in MECH_ORDER],
                    fontsize=11, fontweight="bold")
ax1.set_yticks(range(len(SUBSTRATE_ORDER)))
ax1.set_yticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                    fontsize=11, fontweight="bold")
ax1.set_title("(a) 实验覆盖率（n）", fontsize=12.5, fontweight="bold", pad=10)

for i in range(len(SUBSTRATE_ORDER)):
    for j in range(len(MECH_ORDER)):
        v = n_arr[i, j]
        if np.isnan(v):
            ax1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor="#cccccc", edgecolor="#999999",
                                    lw=1.5, hatch="///"))
            ax1.text(j, i, "n=0", ha="center", va="center",
                     fontsize=10, color="#444", fontweight="bold")
        else:
            txt_color = "white" if v > 300 else "black"
            ax1.text(j, i, f"{int(v)}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=txt_color)
            # 红色边框: n < 10 (实验缺口)
            if v < 10:
                ax1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="#E63946",
                                        lw=3.5))

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("样本数 n", fontsize=10.5, fontweight="bold")

# ---------- (b) 平均产率热图 ----------
yield_arr = pivot_yield.values.astype(float)
im2 = ax2.imshow(np.where(np.isnan(yield_arr), 50, yield_arr),
                 cmap="RdYlGn", aspect="auto", vmin=40, vmax=100)
ax2.set_xticks(range(len(MECH_ORDER)))
ax2.set_xticklabels([MECH_LABEL[m] for m in MECH_ORDER],
                    fontsize=11, fontweight="bold")
ax2.set_yticks(range(len(SUBSTRATE_ORDER)))
ax2.set_yticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                    fontsize=11, fontweight="bold")
ax2.set_title("(b) 平均产率 (%)", fontsize=12.5, fontweight="bold", pad=10)

for i in range(len(SUBSTRATE_ORDER)):
    for j in range(len(MECH_ORDER)):
        v = yield_arr[i, j]
        if np.isnan(v):
            ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor="#cccccc", edgecolor="#999999",
                                    lw=1.5, hatch="///"))
            ax2.text(j, i, "n=0", ha="center", va="center",
                     fontsize=10, color="#444", fontweight="bold")
        else:
            txt_color = "white" if v < 55 else "black"
            ax2.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=txt_color)
            # CHO 行加红色边框强化
            if SUBSTRATE_ORDER[i] == "Cyclohexene oxide":
                ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="#E63946",
                                        lw=3.0))
            # 实验缺口单元
            if not np.isnan(n_arr[i, j]) and n_arr[i, j] < 10:
                ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="#1D3557",
                                        lw=2.0, ls="--"))

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("平均产率 (%)", fontsize=10.5, fontweight="bold")

# legend
red_border = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#E63946", lw=3.0,
                       label="CHO 行 / 红色边框")
dashed = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1D3557", lw=2.0, ls="--",
                   label="n < 10 实验缺口")
fig.legend(handles=[red_border, dashed], loc="lower center",
           ncol=2, fontsize=10.5, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Fig. 6.  5 × 5 底物 × 机制矩阵：覆盖率与平均产率",
             fontsize=14.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.07,
         "数据来源：results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv  |  "
         "n = 2,316  |  CHO 在 5 机制单元均 < 68% (端位 4 底物均 > 75%)；"
         "5 个 n < 10 单元为 Reaxys 已发表但覆盖度极低的实验缺口",
         ha="center", fontsize=9, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("fig6_transferability_matrix.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("fig6_transferability_matrix.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("[Saved] fig6_transferability_matrix.png / .pdf")