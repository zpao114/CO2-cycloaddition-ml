# -*- coding: utf-8 -*-
"""
figures/fig_7_shap_direction.py
==================================
Fig. 7 — Per-substrate SHAP 方向反转图（5 底物 × 5 top 特征）。

主信息：
  (1) CHO 的 top-1 = time_log（动力学），4 个端位底物的 top-1 = sub_homo_eV（电子驱动）—— 类别错位
  (2) sub_homo_eV 在 CHO 上 signed SHAP = -1.20 vs 端位 +2.42 ~ +5.02 —— 同特征符号反转
  (3) delta_E_HL 在 CHO 上 signed SHAP = -1.14 vs 端位 +0.27~+0.47 —— 第二条反转证据

排版：5 子图 × 1 行 × 共享 y 轴，CHO 用红色 #E63946 标注；sub_homo_eV 子图加
     "CHO 反向" 注释，delta_E_HL 子图加 "CHO 反向" 注释。

输入：results_step4_5/per_substrate_top_features.csv
输出：fig7_shap_direction.png / .pdf
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
df = pd.read_csv('results/results_step4_5/per_substrate_top_features.csv')

SUBSTRATE_ORDER = [
    "Cyclohexene oxide",          # CHO — endo- / 红色
    "Propylene oxide",
    "Styrene oxide",
    "Epichlorohydrin",
    "Isopropyl glycidyl ether",
]
SUBSTRATE_LABEL = {
    "Cyclohexene oxide":         "CHO\n(环内)",
    "Propylene oxide":           "PO",
    "Styrene oxide":             "SO",
    "Epichlorohydrin":           "ECH",
    "Isopropyl glycidyl ether":  "IGE",
}
SUBSTRATE_COLOR = {
    "Cyclohexene oxide":         "#E63946",   # 番茄红 — CHO outlier
    "Propylene oxide":           "#457B9D",
    "Styrene oxide":             "#457B9D",
    "Epichlorohydrin":           "#457B9D",
    "Isopropyl glycidyl ether":  "#457B9D",
}

# 5 个特征：CHO top-1 是 time_log, 端位 top-1 是 sub_homo_eV
FEATURE_ORDER = [
    "sub_homo_eV",       # 端位 top-1 (反例)
    "delta_E_HL",        # 第二条反转
    "temperature",
    "pressure",
    "time_log",          # CHO top-1
]
FEATURE_LABEL = {
    "sub_homo_eV":   "sub_homo_eV\n(底物 HOMO)",
    "delta_E_HL":    "ΔE_HL\n(cat HOMO − sub HOMO)",
    "temperature":   "Temperature",
    "pressure":      "Pressure",
    "time_log":      "time_log\n(反应时长)",
}

pivot = (df.pivot_table(index="substrate",
                        columns="feature",
                        values="mean_signed_shap",
                        aggfunc="first")
           .reindex(index=SUBSTRATE_ORDER, columns=FEATURE_ORDER))

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 6.2), sharey=True)
fig.patch.set_facecolor("white")

YMIN, YMAX = -2.6, 6.0

for feat_idx, feat in enumerate(FEATURE_ORDER):
    ax = axes[feat_idx]
    vals = pivot[feat].values
    colors = [SUBSTRATE_COLOR[s] for s in SUBSTRATE_ORDER]

    bars = ax.bar(range(len(SUBSTRATE_ORDER)), vals, 0.62,
                  color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.4)

    ax.axhline(0, color="#222", lw=1.0, ls="-", alpha=0.5)

    ax.set_xticks(range(len(SUBSTRATE_ORDER)))
    ax.set_xticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                       fontsize=10.5, fontweight="bold")
    ax.set_title(FEATURE_LABEL[feat], fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.22, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    if feat_idx == 0:
        ax.set_ylabel("Mean Signed SHAP\n(+ 推高产率, − 推低产率)",
                      fontsize=11, fontweight="bold")

    # value labels
    for bar, v in zip(bars, vals):
        if pd.isna(v):
            continue
        offset = 0.20 if v >= 0 else -0.20
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold", color="#222")

    # 子图标题下加注释
    if feat == "sub_homo_eV":
        # sub_homo_eV: 端位为正 4/4, CHO 为负 — top-1 类别错位的关键证据
        ax.annotate("CHO 唯一反向\n(端位均 +)",
                    xy=(0, vals[0]), xytext=(0.8, vals[0] - 1.6),
                    fontsize=10, color="#E63946", fontweight="bold",
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.8))
    elif feat == "delta_E_HL":
        ax.annotate("CHO 反向",
                    xy=(0, vals[0]), xytext=(0.8, vals[0] - 1.2),
                    fontsize=10, color="#E63946", fontweight="bold",
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.8))
    elif feat == "time_log":
        # CHO top-1 是 time_log — 类别错位的另一面
        ax.annotate("CHO top-1 = time_log\n(动力学控制)",
                    xy=(0, abs(vals[0]) if not pd.isna(vals[0]) else 1.9),
                    xytext=(1.2, 4.6),
                    fontsize=10, color="#E63946", fontweight="bold",
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.8))

# common y limits
for ax in axes:
    ax.set_ylim(YMIN, YMAX)

# legend
red_patch = Rectangle((0, 0), 1, 1, color="#E63946", alpha=0.88,
                      label="CHO（环内环氧化物）")
blue_patch = Rectangle((0, 0), 1, 1, color="#457B9D", alpha=0.88,
                       label="PO / SO / ECH / IGE（端位）")
fig.legend(handles=[red_patch, blue_patch], loc="lower center",
           ncol=2, fontsize=10.5, frameon=True,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Fig. 7.  Per-substrate SHAP 方向诊断：CHO 走独立的反应通道",
             fontsize=14.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.06,
         "证据 1 — top-1 类别错位（CHO = time_log 动力学；端位 = sub_homo_eV 电子驱动）  |  "
         "证据 2 — sub_homo_eV 符号反转（CHO −1.20 vs 端位 +2.42 ~ +5.02）  |  "
         "n = 2,316  |  数据来源：results_step4_5/per_substrate_top_features.csv",
         ha="center", fontsize=9, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("fig7_shap_direction.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("fig7_shap_direction.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("[Saved] fig7_shap_direction.png / .pdf")