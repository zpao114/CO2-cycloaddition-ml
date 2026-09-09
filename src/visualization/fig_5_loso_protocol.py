# -*- coding: utf-8 -*-
"""
figures/fig_5_loso_protocol.py
=================================
Fig. 5 — LOSO / LOMO / LOSO×LOMO 四协议 R² 横向柱状图（XGBoost）。

主信息：
  - 5-fold (内插基线) → R² ≈ +0.30
  - LOMO (跨机制) → R² ≈ +0.15
  - LOSO (跨底物) → R² ≈ -0.05 — **跨底物结构性崩塌**
  - LOSO×LOMO → R² ≈ +0.21 — n 减小带来的 "假性反弹"

排版：单图、4 协议 × 2 特征集 (xTB only / xTB + mechanism)；CHO 红箭头突出 LOSO 失败。

输入：results_step4/summary_protocol.csv
输出：fig5_loso_protocol.png / .pdf
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
df = pd.read_csv('results/results_step4/summary_protocol.csv')

PROTOCOL_LABEL = {
    "5fold":        "5-fold\n(内插基线)",
    "LOMO":         "LOMO\n(跨机制)",
    "LOSO":         "LOSO\n(跨底物)",
    "LOSO×LOMO":    "LOSO × LOMO\n(双重跨)",
}
PROTOCOL_ORDER = ["5fold", "LOMO", "LOSO", "LOSO×LOMO"]

FEATURE_LABEL = {
    "X0_xTB_only":    "X0 (xTB only)",
    "X1_xTB+mech":    "X1 (xTB + mech one-hot)",
}
FEATURE_COLOR = {
    "X0_xTB_only":   "#457B9D",   # 深蓝
    "X1_xTB+mech":   "#E09F3E",   # 暖橙
}
FEATURE_ORDER = ["X0_xTB_only", "X1_xTB+mech"]

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 6.5))
fig.patch.set_facecolor("white")

x = np.arange(len(PROTOCOL_ORDER))
width = 0.36

for f_idx, feat in enumerate(FEATURE_ORDER):
    sub = (df[df["feature_set"] == feat]
           .set_index("protocol")
           .reindex(PROTOCOL_ORDER))
    r2_vals = sub["r2"].values
    offset = (f_idx - 0.5) * width
    bars = ax.bar(x + offset, r2_vals, width,
                  color=FEATURE_COLOR[feat], alpha=0.88,
                  edgecolor="white", linewidth=1.3,
                  label=FEATURE_LABEL[feat])

    # value labels
    for bar, r2 in zip(bars, r2_vals):
        if r2 >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, r2 + 0.018,
                    f"R²={r2:+.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    color=FEATURE_COLOR[feat])
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, r2 - 0.022,
                    f"R²={r2:+.3f}", ha="center", va="top",
                    fontsize=10, fontweight="bold",
                    color=FEATURE_COLOR[feat])

# zero reference
ax.axhline(0, color="#222", lw=1.0, ls="-", alpha=0.6)

# 关键标注: LOSO 跨底物崩塌
loso_idx = PROTOCOL_ORDER.index("LOSO")
ax.annotate("跨底物结构性崩塌\n(LOSO R² = −0.051)",
            xy=(loso_idx - 0.18, -0.051),
            xytext=(loso_idx + 0.4, -0.32),
            fontsize=11.5, color="#E63946", fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#E63946", lw=2.0))

# 5-fold 标注
fold_idx = PROTOCOL_ORDER.index("5fold")
ax.annotate("内插基线\n(R² ≈ 0.30)",
            xy=(fold_idx, 0.30),
            xytext=(fold_idx, 0.45),
            fontsize=10.5, color="#1D3557", fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.5))

# LOSO×LOMO 假性反弹注释
both_idx = PROTOCOL_ORDER.index("LOSO×LOMO")
ax.annotate("n 减小造成的\n'假性反弹' (R² = +0.22)",
            xy=(both_idx + 0.18, 0.21),
            xytext=(both_idx - 0.7, 0.36),
            fontsize=10, color="#777", fontweight="normal",
            ha="center", style="italic",
            arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))

ax.set_xticks(x)
ax.set_xticklabels([PROTOCOL_LABEL[p] for p in PROTOCOL_ORDER],
                   fontsize=12, fontweight="bold")
ax.set_ylabel("R²", fontsize=12.5, fontweight="bold")
ax.set_title("Fig. 5.  LOSO / LOMO 四协议 R²：跨底物崩塌的定量测量",
             fontsize=14.5, fontweight="bold", pad=15)

ax.set_ylim(-0.55, 0.55)
ax.grid(axis="y", alpha=0.22, ls="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=11)
ax.legend(loc="upper right", fontsize=11, frameon=True)

# caption
fig.text(0.5, -0.02,
         "数据来源：results_step4/summary_protocol.csv  |  "
         "模型: XGBoost, n_test = 2,316  |  "
         "X1 (xTB + mech one-hot) 在 LOSO 上仅改善 ΔR² = +0.032，说明崩塌原因在底物维度而非机制维度",
         ha="center", fontsize=9, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("fig5_loso_protocol.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("fig5_loso_protocol.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("[Saved] fig5_loso_protocol.png / .pdf")