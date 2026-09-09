# -*- coding: utf-8 -*-
"""
figures/fig_0_graphical_abstract.py
=====================================
Graphical Abstract（ACS 投稿版，5 × 10 cm 大尺寸）

主图三栏：
  - 左栏：4 阶段清洗漏斗（5,263 → 3,733 → 2,602 → 2,316）+ 数据量标注
  - 中栏：5 类催化系统分布饼图（IL 79.6% / MH 7.6% / Mixed 6.7% / BAS 2.8% / Unknown 3.2%）
  - 右栏：5 × 5 矩阵热图（产率），红色边框标 n < 10 缺口单元，CHO 行加红边框

输入：
  - 数据从已处理 csv 读取
  - 5×5 矩阵使用 results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv
输出：fig0_graphical_abstract.png / .pdf
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

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

# ── 数据 ─────────────────────────────────────────────────────────────────────
# 漏斗：4 阶段
funnel = [
    ("Reaxys 候选",  5263, "#1D3557"),
    ("PDF 解析成功", 3733, "#457B9D"),
    ("产率有效",     2602, "#A8DADC"),
    ("SMILES 标准化", 2316, "#E63946"),  # 终态高亮
]

# 5 类催化系统
cat_dist = pd.DataFrame({
    "name": ["Ionic Liquid\n(NUC)", "Metal Halide\n(LAC)", "Mixed\n(BIF)",
             "Organic Base\n(BAS)", "Unknown\n(OTH)"],
    "n": [1844, 176, 156, 65, 75],
    "color": ["#457B9D", "#E09F3E", "#2A9D8F", "#9C6644", "#777777"],
})
total_cat = cat_dist["n"].sum()
cat_dist["pct"] = cat_dist["n"] / total_cat * 100

# 5×5 矩阵
matrix = pd.read_csv('results/results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv')
SUBSTRATE_ORDER = ["Styrene oxide", "Epichlorohydrin", "Propylene oxide",
                   "Cyclohexene oxide", "Isopropyl glycidyl ether"]
SUBSTRATE_LABEL = {
    "Styrene oxide":            "SO",
    "Epichlorohydrin":          "ECH",
    "Propylene oxide":          "PO",
    "Cyclohexene oxide":        "CHO\n(环内)",
    "Isopropyl glycidyl ether": "IGE",
}
MECH_ORDER = ["ionic_liquid", "metal_halide", "mixed_system",
              "organic_base", "unknown"]
MECH_LABEL = {
    "ionic_liquid":  "IL",
    "metal_halide":  "MH",
    "mixed_system":  "Mixed",
    "organic_base":  "Base",
    "unknown":       "Unk.",
}
pivot_n = matrix.pivot_table(index="reactant", columns="catalyst_family",
                              values="n", aggfunc="first"
                              ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)
pivot_yield = matrix.pivot_table(index="reactant", columns="catalyst_family",
                                   values="yield_mean", aggfunc="first"
                                   ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)

# ── 布局：3 列 × 1 行 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7.5))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.30,
                       left=0.04, right=0.97, top=0.88, bottom=0.05)

# ─── 左栏：漏斗 ─────────────────────────────────────────────────────────────
ax_funnel = fig.add_subplot(gs[0, 0])
ax_funnel.set_xlim(0, 100)
ax_funnel.set_ylim(0, 100)
ax_funnel.axis("off")
ax_funnel.set_title("(左) 数据清洗漏斗\n4 阶段 5,263 → 2,316 反应",
                    fontsize=13, fontweight="bold", pad=12)

y_positions = [85, 65, 45, 22]
widths_pct = [100, 71, 49, 44]  # 比例可视

for i, (label, n, color) in enumerate(funnel):
    y = y_positions[i]
    w = widths_pct[i]
    rect = FancyBboxPatch((50 - w/2, y - 5), w, 12,
                           boxstyle="round,pad=0.4",
                           facecolor=color, edgecolor="white",
                           linewidth=2, alpha=0.92)
    ax_funnel.add_patch(rect)
    text_color = "white" if i < 3 else "white"
    ax_funnel.text(50, y + 1, f"{label}",
                    ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=text_color)
    ax_funnel.text(50, y - 3, f"n = {n:,}",
                    ha="center", va="center",
                    fontsize=10.5, color="white")

# 箭头
for i in range(3):
    y_top = y_positions[i] - 5
    y_bot = y_positions[i + 1] + 7
    ax_funnel.annotate("", xy=(50, y_bot), xytext=(50, y_top),
                       arrowprops=dict(arrowstyle="->", color="#444",
                                       lw=1.6))
    # 丢弃原因
    drops = ["(PDF 解析失败)", "(无效产率)", "(SMILES 标准化失败)"][i]
    ax_funnel.text(85, (y_top + y_bot) / 2, drops,
                    fontsize=9.5, color="#666", style="italic",
                    ha="left", va="center")

# ─── 中栏：催化系统饼图 ────────────────────────────────────────────────────
ax_pie = fig.add_subplot(gs[0, 1])
ax_pie.set_title("(中) 催化系统分布\nn = 2,316 反应",
                 fontsize=13, fontweight="bold", pad=12)

wedges, texts, autotexts = ax_pie.pie(
    cat_dist["n"], labels=cat_dist["name"],
    colors=cat_dist["color"], autopct=lambda p: f"{p:.1f}%",
    startangle=90, pctdistance=0.72,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10.5, fontweight="bold"),
)
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
    autotext.set_fontsize(10.5)

# ─── 右栏：5×5 矩阵热图 ────────────────────────────────────────────────────
ax_mat = fig.add_subplot(gs[0, 2])
ax_mat.set_title("(右) 5 × 5 底物 × 催化体系矩阵\n平均产率 (%) — 红框 = 实验缺口",
                 fontsize=13, fontweight="bold", pad=12)

yield_arr = pivot_yield.values.astype(float)
n_arr = pivot_n.values.astype(float)
disp = np.where(np.isnan(yield_arr), 50, yield_arr)
im = ax_mat.imshow(disp, cmap="RdYlGn", aspect="auto", vmin=40, vmax=100)
ax_mat.set_xticks(range(len(MECH_ORDER)))
ax_mat.set_xticklabels([MECH_LABEL[m] for m in MECH_ORDER],
                        fontsize=10.5, fontweight="bold")
ax_mat.set_yticks(range(len(SUBSTRATE_ORDER)))
ax_mat.set_yticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                        fontsize=10.5, fontweight="bold")

for i in range(len(SUBSTRATE_ORDER)):
    for j in range(len(MECH_ORDER)):
        v = yield_arr[i, j]
        n = n_arr[i, j]
        if np.isnan(v):
            ax_mat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor="#cccccc", edgecolor="#999",
                                        lw=1.5, hatch="///"))
            ax_mat.text(j, i, "n=0", ha="center", va="center",
                        fontsize=10, color="#444", fontweight="bold")
        else:
            txt_color = "white" if v < 55 else "black"
            ax_mat.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=txt_color)
            # 红色边框：n < 10
            if not np.isnan(n) and n < 10:
                ax_mat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#E63946",
                                            lw=2.5))
            # CHO 行加红边框
            if SUBSTRATE_ORDER[i] == "Cyclohexene oxide":
                ax_mat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#1D3557",
                                            lw=2.0))

cbar = plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
cbar.set_label("平均产率 (%)", fontsize=10.5, fontweight="bold")

# ── 主标题 ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Graphical Abstract  ·  CO₂ 环加成 ML 跨底物诊断  ·  "
    "LOSO 失败根因 (CHO R² = −1.45)  ·  5×5 矩阵标注 4 个实验缺口",
    fontsize=15, fontweight="bold", y=0.98)

# ── 底部 caption ────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "2,316 reactions  ·  5 epoxides (PO/ECH/SO/CHO/IGE)  ·  "
         "5 catalyst families  ·  Reaxys 2026  ·  "
         "n < 10 单元 (红框) = Reaxys 已发表但覆盖度极低  ·  CHO 行深蓝边框 = 环内环氧化物独立通道",
         ha="center", fontsize=10, color="#444", style="italic")

plt.savefig("fig0_graphical_abstract.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("fig0_graphical_abstract.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("[Saved] fig0_graphical_abstract.png / .pdf")