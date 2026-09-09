# -*- coding: utf-8 -*-
"""
figures/fig_toc.py
====================
TOC Graphic（ACS 投稿版 3.5 × 8.5 cm @ 600 dpi）

构图（v4，3 栏）：
  (a) 左栏：5 × 5 矩阵（产率 + 缺口标注）
  (b) 中栏：per-substrate SHAP 方向热图重绘（5 × 3：top-3 特征 × 5 底物）
  (c) 右栏：LOSO 预测偏差柱图（5 底物，CHO 红框 +34.6%）

主标题（顶部黄色 banner）：
  "LOSO Diagnostics · CHO Root Cause · 5×5 Gaps & Priorities"

输入：
  - results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv
  - results_step4_5/per_substrate_top_features.csv
  - results_step7_improved_loso/loso_per_substrate_bias_summary.csv
输出：fig_toc.png / .pdf
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
from matplotlib.gridspec import GridSpec

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
# 5×5 矩阵
matrix = pd.read_csv('results/results_substrate_catalyst_matrix/matrix_yield_ci_5x5.csv')
SUBSTRATE_ORDER = ["Styrene oxide", "Epichlorohydrin", "Propylene oxide",
                   "Cyclohexene oxide", "Isopropyl glycidyl ether"]
SUBSTRATE_LABEL = {
    "Styrene oxide":            "SO",
    "Epichlorohydrin":          "ECH",
    "Propylene oxide":          "PO",
    "Cyclohexene oxide":        "CHO",
    "Isopropyl glycidyl ether": "IGE",
}
MECH_ORDER = ["ionic_liquid", "metal_halide", "mixed_system",
              "organic_base", "unknown"]
MECH_LABEL = {
    "ionic_liquid":  "IL", "metal_halide":  "MH", "mixed_system":  "Mix",
    "organic_base":  "Bas", "unknown":      "Unk",
}
pivot_n = matrix.pivot_table(index="reactant", columns="catalyst_family",
                              values="n", aggfunc="first"
                              ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)
pivot_yield = matrix.pivot_table(index="reactant", columns="catalyst_family",
                                   values="yield_mean", aggfunc="first"
                                   ).reindex(index=SUBSTRATE_ORDER, columns=MECH_ORDER)

# SHAP top-3 per substrate
shap = pd.read_csv('results/results_step4_5/per_substrate_top_features.csv')
shap["abs_shap"] = shap["mean_signed_shap"].abs()
shap_top3 = (shap.sort_values(["substrate", "abs_shap"], ascending=[True, False])
                .groupby("substrate").head(3))

# LOSO bias
bias = pd.read_csv('results/results_step7_improved_loso/loso_per_substrate_bias_summary.csv')
bias = bias.rename(columns={
    "n": "n_test",
    "actual_yield_mean_pct": "true_mean",
    "predicted_yield_mean_pct": "pred_mean",
    "prediction_bias_pct": "bias_pct",
})
bias_sorted = bias.set_index("substrate").reindex(SUBSTRATE_ORDER)

# ── Figure ──────────────────────────────────────────────────────────────────
# ACS TOC 规范: 3.5 × 8.5 cm @ 600 dpi → 实际绘制宽高比 ~1:2.5
# 横向 3 栏，整体高度较小
fig = plt.figure(figsize=(12, 4.5))
fig.patch.set_facecolor("white")

# 顶部黄色 banner
banner = FancyBboxPatch((0.02, 0.86), 0.96, 0.11,
                         boxstyle="round,pad=0.005",
                         facecolor="#FAD02E", edgecolor="#444", lw=0.8,
                         transform=fig.transFigure)
fig.patches.append(banner)
fig.text(0.5, 0.915,
         "LOSO Diagnostics  ·  CHO Root Cause  ·  5×5 Gaps & Priorities",
         ha="center", va="center", fontsize=12, fontweight="bold",
         color="#1D3557")

# 3 栏
gs = GridSpec(1, 3, left=0.04, right=0.97, top=0.82, bottom=0.06,
               wspace=0.30, width_ratios=[1.0, 1.0, 1.0])

# ─── (a) 左栏: 5×5 矩阵 ──────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_title("(a) 5×5 矩阵\n产率 + 缺口标注",
                fontsize=10, fontweight="bold", pad=6)
yield_arr = pivot_yield.values.astype(float)
n_arr = pivot_n.values.astype(float)
disp = np.where(np.isnan(yield_arr), 50, yield_arr)
im = ax_a.imshow(disp, cmap="RdYlGn", aspect="auto", vmin=40, vmax=100)
ax_a.set_xticks(range(len(MECH_ORDER)))
ax_a.set_xticklabels([MECH_LABEL[m] for m in MECH_ORDER],
                      fontsize=9, fontweight="bold")
ax_a.set_yticks(range(len(SUBSTRATE_ORDER)))
ax_a.set_yticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                      fontsize=9, fontweight="bold")
for i in range(len(SUBSTRATE_ORDER)):
    for j in range(len(MECH_ORDER)):
        v = yield_arr[i, j]
        n = n_arr[i, j]
        if np.isnan(v):
            ax_a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#cccccc", edgecolor="#999",
                                       lw=1.0, hatch="///"))
            ax_a.text(j, i, "0", ha="center", va="center",
                       fontsize=8, color="#444")
        else:
            txt_color = "white" if v < 55 else "black"
            ax_a.text(j, i, f"{v:.0f}", ha="center", va="center",
                       fontsize=9, fontweight="bold", color=txt_color)
            if not np.isnan(n) and n < 10:
                ax_a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          fill=False, edgecolor="#E63946",
                                          lw=2.0))
            if SUBSTRATE_ORDER[i] == "Cyclohexene oxide":
                ax_a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          fill=False, edgecolor="#1D3557",
                                          lw=1.6))
plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04).set_label("Yield (%)",
                                                                  fontsize=8)

# ─── (b) 中栏: SHAP top-3 重绘热图 ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title("(b) Per-substrate SHAP\nTop-3 signed values",
                fontsize=10, fontweight="bold", pad=6)
# 构 5 (substrate) × 3 (top feature) 的 signed SHAP 矩阵
shap_pivot = (shap_top3.pivot_table(index="substrate",
                                      columns="feature",
                                      values="mean_signed_shap",
                                      aggfunc="first")
                 .reindex(index=SUBSTRATE_ORDER))
shap_pivot = shap_pivot.fillna(0)
im_b = ax_b.imshow(shap_pivot.values, cmap="RdBu_r", aspect="auto",
                    vmin=-3, vmax=6)
ax_b.set_xticks(range(len(shap_pivot.columns)))
ax_b.set_xticklabels([c.replace("sub_homo_eV", "HOMO")
                       .replace("delta_E_HL", "ΔE_HL")
                       .replace("delta_E_LL", "ΔE_LL")
                       .replace("time_log", "t_log")
                       .replace("temperature", "T")
                       .replace("pressure", "P")
                       .replace("sub_lumo_eV", "LUMO") for c in shap_pivot.columns],
                      fontsize=8.5, fontweight="bold", rotation=30, ha="right")
ax_b.set_yticks(range(len(SUBSTRATE_ORDER)))
ax_b.set_yticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                      fontsize=9, fontweight="bold")
for i in range(len(SUBSTRATE_ORDER)):
    for j in range(len(shap_pivot.columns)):
        v = shap_pivot.values[i, j]
        txt_color = "white" if abs(v) > 3 else "black"
        ax_b.text(j, i, f"{v:+.1f}" if v != 0 else "·",
                   ha="center", va="center",
                   fontsize=8, fontweight="bold", color=txt_color)
        # CHO 行加黑边
        if SUBSTRATE_ORDER[i] == "Cyclohexene oxide":
            ax_b.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2.0))
plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04).set_label("Signed SHAP",
                                                                     fontsize=8)

# ─── (c) 右栏: LOSO 偏差柱图 ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_title("(c) LOSO 预测偏差\n(实际 − 预测, %)",
                fontsize=10, fontweight="bold", pad=6)
bias_vals = bias_sorted["bias_pct"].values
colors_c = ["#E63946" if s == "Cyclohexene oxide" else "#457B9D"
             for s in SUBSTRATE_ORDER]
bars = ax_c.bar(range(len(SUBSTRATE_ORDER)), bias_vals, 0.62,
                  color=colors_c, alpha=0.88,
                  edgecolor="white", linewidth=1.0)
ax_c.axhline(0, color="#222", lw=0.8, ls="-", alpha=0.5)
ax_c.set_xticks(range(len(SUBSTRATE_ORDER)))
ax_c.set_xticklabels([SUBSTRATE_LABEL[s] for s in SUBSTRATE_ORDER],
                      fontsize=9, fontweight="bold")
ax_c.set_ylabel("偏差 (%)", fontsize=9, fontweight="bold")
ax_c.grid(axis="y", alpha=0.22, ls="--")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
for bar, v in zip(bars, bias_vals):
    offset = 2.5 if v >= 0 else -2.5
    ax_c.text(bar.get_x() + bar.get_width() / 2, v + offset,
               f"{v:+.1f}", ha="center",
               va="bottom" if v >= 0 else "top",
               fontsize=8, fontweight="bold")
# CHO 标注
cho_idx = SUBSTRATE_ORDER.index("Cyclohexene oxide")
ax_c.annotate("CHO: +34.6%\nR² = −1.45",
              xy=(cho_idx, bias_vals[cho_idx]),
              xytext=(cho_idx - 1.4, -30),
              fontsize=8.5, color="#E63946", fontweight="bold", ha="center",
              arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2))

plt.savefig("fig_toc.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig("fig_toc.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("[Saved] fig_toc.png / .pdf")