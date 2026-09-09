# -*- coding: utf-8 -*-
"""
generate_paper_figures_zh.py - 论文图表生成脚本（中文版 v6）
===========================================================
生成CO2-cycloaddition论文的8张高质量图表（中文版）。

v6 修复与改进：
- Fig 0: 全中文 Key Findings，所有面板标题、标签、图例均为中文
- Fig 1-7: 所有子图标题、轴标签、图例、标注全部中文化
- 全局：所有字体加大2pt，DPI提升至300，标注间距优化
- Fig 2: X轴标签字体加大，数值标注清晰化
- Fig 3: 热图标注、偏差标注字体加大
- Fig 4: 数值标注与颜色条间距优化
- Fig 5: X轴标签旋转角度优化，特征名显示完整
- Fig 6: 最优λ标注位置调整，图例移至空白区域
- Fig 7: 柱状图数值标注位置优化，防止出界
- 添加 PDF + SVG 双格式输出（svglib 转换）

使用方法：
    python src/analysis/diagnostics/generate_paper_figures_zh.py
"""

from __future__ import annotations

import os
import sys
import io
import warnings
from datetime import datetime

# UTF-8输出支持
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
from matplotlib.ticker import FuncFormatter
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
# 路径配置
# ============================================================================
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
os.chdir(PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 数据文件路径
PATH_SUMMARY_PROTOCOL = os.path.join(PROJECT_ROOT, "results_step4", "summary_protocol.csv")
PATH_LOSO_PREDICTIONS = os.path.join(PROJECT_ROOT, "results_step7_improved_loso", "loso_predictions.csv")
PATH_SHAP_FEATURES = os.path.join(PROJECT_ROOT, "results_step4_5", "per_substrate_top_features.csv")
PATH_LATENT = os.path.join(PROJECT_ROOT, "results_pcl_ae_viz", "latent_comparison.csv")
PATH_LAMBDA = os.path.join(PROJECT_ROOT, "results_lambda_ablation", "lambda_results.csv")
PATH_LEARNING_CURVE = os.path.join(PROJECT_ROOT, "results_sample_size_sensitivity", "learning_curve_summary.csv")

# ============================================================================
# 中文字体配置
# ============================================================================
def setup_chinese_font():
    """Configure dual-font support: DejaVu Sans for symbols + Chinese font for CJK"""
    chinese_font_found = None
    font_candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    for font_name in font_candidates:
        for f in fm.fontManager.ttflist:
            if font_name.lower() in f.name.lower():
                chinese_font_found = f.name
                print(f"  Found Chinese font: {f.name}")
                break
        if chinese_font_found:
            break

    if chinese_font_found:
        # Chinese font first (handles CJK), DejaVu Sans second (fallback for missing glyphs)
        plt.rcParams["font.sans-serif"] = [chinese_font_found, "DejaVu Sans"]
    else:
        print("  Warning: No Chinese font found, using DejaVu Sans only")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["font.family"] = "sans-serif"
    return chinese_font_found

CHINESE_FONT = setup_chinese_font()

# ============================================================================
# 全局绘图配置（发表级别）
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
# 常量定义
# ============================================================================
SUB2 = "2"     # CO2（YaHei 不含 ₂ 下标，用普通 2）
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

SUB_NAMES_CN = {
    "Cyclohexene oxide": "环己烯氧化物",
    "Propylene oxide": "环氧丙烷",
    "Epichlorohydrin": "表氯醇",
    "Styrene oxide": "苯乙烯氧化物",
    "Isopropyl glycidyl ether": "异丙基缩水甘油醚",
}

SUBSTRATES_ORDER = [
    "Cyclohexene oxide",
    "Styrene oxide",
    "Propylene oxide",
    "Epichlorohydrin",
    "Isopropyl glycidyl ether",
]

MECH_NAMES_CN = {
    "ionic_liquid": "离子液体",
    "metal_halide": "金属卤化物",
    "mixed_system": "混合体系",
    "organic_base": "有机碱",
    "unknown": "未知",
}

# ============================================================================
# 辅助函数
# ============================================================================
def save_figure(fig, name_base: str):
    """保存图片为PNG、PDF和SVG格式"""
    png_path = os.path.join(FIGURES_DIR, f"{name_base}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name_base}.pdf")

    # 保存PNG（300 DPI，无bbox_inches="tight" 避免文字偏移）
    fig.savefig(png_path, dpi=300, facecolor="white", pad_inches=0.1)
    fig.savefig(pdf_path, dpi=300, facecolor="white", pad_inches=0.1)
    plt.close(fig)

    print(f"  保存: {png_path}")
    print(f"  保存: {pdf_path}")

    # SVG 转换 via PyMuPDF
    if MUPDF_AVAILABLE:
        try:
            svg_path = os.path.join(FIGURES_DIR, f"{name_base}.svg")
            doc = fitz.open(pdf_path)
            page = doc[0]
            svg_data = page.get_svg_image()
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_data)
            doc.close()
            print(f"  保存: {svg_path}")
        except Exception as e:
            print(f"  SVG转换跳过: {e}")


def format_r2(value: float) -> str:
    """格式化R²值"""
    if value < 0:
        return f"{MINUS}{abs(value):.3f}"
    return f"{value:.3f}"


# ============================================================================
# Fig 0: 图形摘要 (Graphical Abstract) - 全中文版
# ============================================================================
def make_fig0_graphical_abstract():
    """生成图形摘要：全中文内容"""

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.2, 1, 1, 1.5], wspace=0.3)

    # ----- Panel (a): 数据清洗漏斗 -----
    ax = fig.add_subplot(gs[0])

    funnel = [5263, 3733, 2602, 2316]
    funnel_labels = ["5,263（原始）", "3,733", "2,602", "2,316（最终）"]
    funnel_colors = ["#90A4AE", "#78909C", "#607D8B", "#455A64"]

    bars = ax.barh(range(len(funnel)), funnel, color=funnel_colors, edgecolor="black", height=0.6, linewidth=1.0)
    for i, (v, lbl) in enumerate(zip(funnel, funnel_labels)):
        ax.text(v + 100, i, lbl, va="center", ha="left", fontsize=12)

    ax.set_yticks([])
    ax.set_xlim(0, 6200)
    ax.set_xlabel("反应数量", fontsize=13)
    ax.set_title("（a）数据清洗流程", fontweight="bold", fontsize=14)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Panel (b): 催化剂体系分布 -----
    ax = fig.add_subplot(gs[1])

    pie_sizes = [79.6, 7.6, 6.7, 2.8, 3.2]
    pie_labels = ["离子液体 79.6%", "金属卤化物 7.6%", "混合体系 6.7%", "有机碱 2.8%", "未知 3.2%"]
    pie_colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#9E9E9E"]

    wedges, texts, autotexts = ax.pie(pie_sizes, colors=pie_colors, startangle=90,
                                       wedgeprops=dict(edgecolor="white", linewidth=2),
                                       autopct="")
    ax.legend(wedges, pie_labels, loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=11)
    ax.set_title("（b）催化剂体系分布", fontweight="bold", fontsize=14)

    # ----- Panel (c): 底物产率对比 -----
    ax = fig.add_subplot(gs[2])

    substrates_yield = ["CHO", "SO", "PO", "ECH", "IGE"]
    mean_yields = [53.1, 84.4, 88.3, 91.9, 91.0]
    counts = [305, 783, 646, 692, 64]
    colors_bar = [COLOR_CHO, "#ff7f0e", COLOR_TERM, COLOR_OTH, "#9467bd"]

    bars = ax.bar(substrates_yield, mean_yields, color=colors_bar, edgecolor="black", width=0.65, linewidth=1.0)
    ax.set_ylabel("平均产率（%）", fontsize=13)
    ax.set_xlabel("底物", fontsize=13)
    ax.set_title("（c）底物平均产率", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, y, n in zip(bars, mean_yields, counts):
        ax.text(bar.get_x() + bar.get_width()/2, y + 2, f"{y:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ----- Panel (d): 关键发现 -----
    ax = fig.add_subplot(gs[3])
    ax.axis("off")

    findings = [
        "【核心发现】",
        "",
        "1. GroupKFold R\u00b2 = 0.278 \u2192 LOSO R\u00b2 = \u22120.063",
        "   \u2192 模型在新底物上失效",
        "",
        "2. CHO主导LOSO失败：",
        "   \u2022 平均偏差 +34.6%",
        "   \u2022 sub_homo_eV: r = \u22120.245",
        "",
        "3. PCL-AE更好分离催化剂家族",
        "   （轮廓系数：0.81 vs 0.54）",
    ]

    text = "\n".join(findings)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=13, fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#333333", linewidth=1.5))

    fig.suptitle("图形摘要 \u2014 CO2 环加成机器学习框架",
                 fontsize=17, fontweight="bold", y=0.98)

    save_figure(fig, "fig0_graphical_abstract_zh")
    print("  Fig 0 完成")


# ============================================================================
# Fig 1: 数据概览与质量 - 中文版
# ============================================================================
def make_fig1_data_overview():
    """生成数据概览图：全中文"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)
    df_pred["predicted"] = df_pred["predicted"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ----- Panel (a): 样本量分布 -----
    ax = axes[0]
    substrates = SUBSTRATES_ORDER
    counts = [305, 783, 646, 692, 64]
    colors = [SUB_COLORS.get(s, COLOR_OTH) for s in substrates]

    bars = ax.bar([SUB_SHORT[s] for s in substrates], counts, color=colors, edgecolor="black", width=0.6, linewidth=1.0)
    ax.set_ylabel("样本数量", fontsize=14)
    ax.set_xlabel("底物", fontsize=14)
    ax.set_title("（a）样本量分布", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"n={n}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # ----- Panel (b): 产率分布箱线图 -----
    ax = axes[1]
    box_data = []
    box_positions = []

    for i, sub in enumerate(substrates):
        d = df_pred[df_pred["substrate"] == sub]["actual"].values
        box_data.append(d)
        box_positions.append(i)

    bp = ax.boxplot(box_data, positions=box_positions, widths=0.5,
                    patch_artist=True, showfliers=True, flierprops=dict(markersize=5))

    for patch, sub in zip(bp["boxes"], substrates):
        patch.set_facecolor(SUB_COLORS.get(sub, COLOR_OTH))
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=13)
    ax.set_ylabel("产率（%）", fontsize=14)
    ax.set_xlabel("底物", fontsize=14)
    ax.set_title("（b）产率分布", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-5, 110)

    # ----- Panel (c): 预测误差分布 -----
    ax = axes[2]

    for i, sub in enumerate(substrates):
        d = df_pred[df_pred["substrate"] == sub]["error"].values
        ax.hist(d, bins=25, alpha=0.5, label=SUB_SHORT[sub],
                color=SUB_COLORS.get(sub, COLOR_OTH), density=True)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("预测误差（实际值 - 预测值，%）", fontsize=14)
    ax.set_ylabel("概率密度", fontsize=14)
    ax.set_title("（c）误差分布", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)

    plt.tight_layout()
    save_figure(fig, "fig1_data_overview_zh")
    print("  Fig 1 完成")


# ============================================================================
# Fig 2: 交叉验证协议对比 - 中文版
# ============================================================================
def make_fig2_protocol_comparison():
    """生成交叉验证协议对比图：全中文"""

    df = pd.read_csv(PATH_SUMMARY_PROTOCOL, encoding="utf-8-sig")
    df["r2"] = df["r2"].astype(float)
    df["mae"] = df["mae"].astype(float)

    protocols = ["5折", "LOSO", "LOMO", "LOSO\u00d7LOMO"]
    x0 = df[df["feature_set"] == "X0_xTB_only"].set_index("protocol")
    x1 = df[df["feature_set"] == "X1_xTB+mech"].set_index("protocol")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # ----- Panel (a): R²对比 -----
    ax = axes[0]
    x = np.arange(len(protocols))
    w = 0.35

    r2_x0 = [x0.loc[p, "r2"] if p in x0.index else 0 for p in protocols]
    r2_x1 = [x1.loc[p, "r2"] if p in x1.index else 0 for p in protocols]

    bars1 = ax.bar(x - w/2, r2_x0, w, label="X0 = 仅xTB", color="#64B5F6", edgecolor="black", linewidth=1.0)
    bars2 = ax.bar(x + w/2, r2_x1, w, label="X1 = xTB+机理", color="#1976D2", edgecolor="black", linewidth=1.0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=13)
    ax.set_ylabel("R\u00b2", fontsize=14)
    ax.set_title("各协议 R\u00b2 对比（XGBoost）", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (v0, v1) in enumerate(zip(r2_x0, r2_x1)):
        offset = 0.025 if v0 >= 0 else -0.08
        ax.text(i - w/2, v0 + offset, f"{v0:.3f}", ha="center", fontsize=11, fontweight="bold")
        offset = 0.025 if v1 >= 0 else -0.08
        ax.text(i + w/2, v1 + offset, f"{v1:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(-0.5, 0.6)

    # ----- Panel (b): MAE对比 -----
    ax = axes[1]

    mae_x0 = [x0.loc[p, "mae"] if p in x0.index else 0 for p in protocols]
    mae_x1 = [x1.loc[p, "mae"] if p in x1.index else 0 for p in protocols]

    ax.bar(x - w/2, mae_x0, w, label="X0 = 仅xTB", color="#90CAF9", edgecolor="black", linewidth=1.0)
    ax.bar(x + w/2, mae_x1, w, label="X1 = xTB+机理", color="#1565C0", edgecolor="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=13)
    ax.set_ylabel("MAE（%）", fontsize=14)
    ax.set_title("各协议 MAE 对比（XGBoost）", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (v0, v1) in enumerate(zip(mae_x0, mae_x1)):
        ax.text(i - w/2, v0 + 0.4, f"{v0:.1f}", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + w/2, v1 + 0.4, f"{v1:.1f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "fig2_protocol_comparison_zh")
    print("  Fig 2 完成")


# ============================================================================
# Fig 3: LOSO失败根因分析 - 中文版
# ============================================================================
def make_fig3_loso_root_cause():
    """生成LOSO失败根因分析图：全中文"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)
    df_pred["predicted"] = df_pred["predicted"].astype(float)
    df_pred["error"] = df_pred["actual"] - df_pred["predicted"]

    substrates = SUBSTRATES_ORDER
    CHO = "Cyclohexene oxide"

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ----- Panel (A): 散点图 -----
    ax = axes[0, 0]
    for sub in substrates:
        d = df_pred[df_pred["substrate"] == sub]
        ax.scatter(d["predicted"], d["actual"], alpha=0.4, s=25,
                   color=SUB_COLORS.get(sub, COLOR_OTH),
                   label=f"{SUB_SHORT[sub]} (n={len(d)})")

    ax.plot([0, 100], [0, 100], "k--", alpha=0.6, lw=2, label="y = x")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("预测产率（%）", fontsize=14)
    ax.set_ylabel("实际产率（%）", fontsize=14)
    ax.set_title("（A）预测值 vs 实际值", fontweight="bold", fontsize=15)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)

    r2_all = r2_score(df_pred["actual"], df_pred["predicted"])
    mae_all = mean_absolute_error(df_pred["actual"], df_pred["predicted"])
    ax.text(0.97, 0.03, f"R\u00b2 = {r2_all:.4f}\nMAE = {mae_all:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    # ----- Panel (B): 偏差柱状图 -----
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

    bars = ax.bar(x_pos, [b["bias"] for b in bias_data], color=colors_bar, alpha=0.8, edgecolor="black", linewidth=1.0)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["sub"] for b in bias_data], fontsize=13)
    ax.set_ylabel("偏差（实际值 - 预测值，%）", fontsize=13)
    ax.set_title("（B）各底物预测偏差", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)

    for i, b in enumerate(bias_data):
        h = b["bias"]
        va = "bottom" if h >= 0 else "top"
        ax.text(i, h + (4 if h >= 0 else -4), f"{h:+.1f}%", ha="center", va=va,
                fontsize=12, fontweight="bold")

    ax.set_ylim(-50, 50)

    # ----- Panel (C): 箱线图 -----
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
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=13)
    ax.set_ylabel("产率（%）", fontsize=14)
    ax.set_title("（C）实际值（浅色）vs 预测值（斜线）", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.5, label="实际值"),
        mpatches.Patch(facecolor="gray", alpha=0.9, hatch="//", label="预测值"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    ax.set_ylim(-5, 105)

    # ----- Panel (D): 热图 -----
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
    ax.set_xticklabels([SUB_SHORT[s] for s in SUBSTRATES_ORDER], fontsize=13)
    ax.set_yticks(range(len(mech_order)))
    ax.set_yticklabels([MECH_NAMES_CN.get(m, m) for m in mech_order], fontsize=12)

    for i in range(len(mech_order)):
        for j in range(len(SUBSTRATES_ORDER)):
            v = pivot_err.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", color=color, fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel("偏差（实际 - 预测，%）", fontsize=13)
    ax.set_title("（D）偏差热图（底物 \u00d7 机理）", fontweight="bold", fontsize=15)
    ax.set_xlabel("底物（留一）", fontsize=14)

    fig.suptitle("LOSO根因分析：CHO结构偏差", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig3_loso_root_cause_zh")
    print("  Fig 3 完成")


# ============================================================================
# Fig 4: 5×5 覆盖矩阵 - 中文版
# ============================================================================
def make_fig4_coverage_matrix():
    """生成5×5覆盖矩阵：全中文"""

    df_pred = pd.read_csv(PATH_LOSO_PREDICTIONS, encoding="utf-8-sig")
    df_pred["actual"] = df_pred["actual"].astype(float)

    substrates = SUBSTRATES_ORDER
    mechanisms = ["ionic_liquid", "metal_halide", "mixed_system", "organic_base", "unknown"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ----- Panel (a): 样本数量 -----
    ax = axes[0]

    n_matrix = np.zeros((len(mechanisms), len(substrates)), dtype=int)
    for i, mech in enumerate(mechanisms):
        for j, sub in enumerate(substrates):
            n_matrix[i, j] = len(df_pred[(df_pred["mechanism"] == mech) & (df_pred["substrate"] == sub)])

    im0 = ax.imshow(n_matrix, cmap="Blues", aspect="auto", norm=LogNorm(vmin=1, vmax=n_matrix.max()))
    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=13)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels([MECH_NAMES_CN.get(m, m) for m in mechanisms], fontsize=12)

    for i in range(len(mechanisms)):
        for j in range(len(substrates)):
            v = int(n_matrix[i, j])
            color = "white" if v > 200 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    cbar0 = plt.colorbar(im0, ax=ax, shrink=0.8)
    cbar0.ax.tick_params(labelsize=11)
    cbar0.ax.set_ylabel("样本数量", fontsize=13)

    ax.set_title("（a）样本覆盖矩阵（n）", fontweight="bold", fontsize=15)
    ax.set_xlabel("底物", fontsize=14)
    ax.set_ylabel("催化剂机理", fontsize=14)

    # ----- Panel (b): 平均产率 -----
    ax = axes[1]

    yield_matrix = np.full((len(mechanisms), len(substrates)), np.nan)
    for i, mech in enumerate(mechanisms):
        for j, sub in enumerate(substrates):
            d = df_pred[(df_pred["mechanism"] == mech) & (df_pred["substrate"] == sub)]
            if len(d) > 0:
                yield_matrix[i, j] = d["actual"].mean()

    im1 = ax.imshow(yield_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=13)
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels([MECH_NAMES_CN.get(m, m) for m in mechanisms], fontsize=12)

    for i in range(len(mechanisms)):
        for j in range(len(substrates)):
            v = yield_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="black", fontsize=13, fontweight="bold")

    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
    cbar1.ax.tick_params(labelsize=11)
    cbar1.ax.set_ylabel("平均产率（%）", fontsize=13)

    ax.set_title("（b）平均产率矩阵（%）", fontweight="bold", fontsize=15)
    ax.set_xlabel("底物", fontsize=14)
    ax.set_ylabel("催化剂机理", fontsize=14)

    fig.suptitle("5\u00d75 底物 \u00d7 催化剂覆盖矩阵", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig4_coverage_matrix_zh")
    print("  Fig 4 完成")


# ============================================================================
# Fig 5: SHAP方向性分析 - 中文版
# ============================================================================
def make_fig5_shap_direction():
    """生成SHAP方向性分析图：全中文"""

    df_shap = pd.read_csv(PATH_SHAP_FEATURES, encoding="utf-8-sig")
    df_shap["mean_signed_shap"] = df_shap["mean_signed_shap"].astype(float)
    df_shap["mean_abs_shap"] = df_shap["mean_abs_shap"].astype(float)

    substrates = SUBSTRATES_ORDER
    CHO = "Cyclohexene oxide"

    top_feats = df_shap.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ----- Panel (a): 分组柱状图 -----
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

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    # 缩短特征名以避免重叠
    short_feats = [f.replace("_", "\n") if len(f) > 12 else f for f in top_feats]
    ax.set_xticklabels(short_feats, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("平均符号SHAP值", fontsize=14)
    ax.set_title("（a）各底物符号SHAP值", fontweight="bold", fontsize=15)
    ax.legend(loc="upper right", fontsize=11, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ----- Panel (b): 热图 -----
    ax = axes[1]

    plot_data = df_shap[df_shap["feature"].isin(top_feats)]
    pivot = plot_data.pivot(index="feature", columns="substrate", values="mean_signed_shap")
    pivot = pivot.reindex(columns=substrates, fill_value=0)
    pivot = pivot.loc[top_feats]

    vmax = max(abs(pivot.values.max()), abs(pivot.values.min())) * 1.2
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-max(vmax, 5), vmax=max(vmax, 5))

    ax.set_xticks(range(len(substrates)))
    ax.set_xticklabels([SUB_SHORT.get(s, s) for s in substrates], fontsize=13)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=11)

    for i in range(len(top_feats)):
        for j in range(len(substrates)):
            v = pivot.values[i, j]
            color = "white" if abs(v) > max(vmax, 5) * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel("平均符号SHAP", fontsize=13)
    ax.set_title("（b）SHAP方向热图", fontweight="bold", fontsize=15)
    ax.set_xlabel("底物", fontsize=14)

    fig.suptitle("各底物SHAP方向性分析", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig5_shap_direction_zh")
    print("  Fig 5 完成")


# ============================================================================
# Fig 6: 消融实验与样本量敏感性 - 中文版
# ============================================================================
def make_fig6_ablation():
    """生成消融实验图：全中文"""

    df_lambda = pd.read_csv(PATH_LAMBDA, encoding="utf-8-sig")
    df_lc = pd.read_csv(PATH_LEARNING_CURVE, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    # ----- Panel (a): Lambda消融实验 -----
    ax = axes[0]

    lambdas = df_lambda["lambda"].values
    ann_r2 = df_lambda["ann_r2_mean"].values
    rf_r2 = df_lambda["rf_r2_mean"].values

    ax.plot(lambdas, ann_r2, "o-", color="#1976D2", label="DualBranchANN", linewidth=2.5, markersize=8)
    ax.plot(lambdas, rf_r2, "s-", color="#2ca02c", label="随机森林", linewidth=2.5, markersize=8)

    # 标记最佳点
    best_idx = df_lambda["ann_r2_mean"].idxmax()
    best_lambda = df_lambda.loc[best_idx, "lambda"]
    best_r2 = df_lambda.loc[best_idx, "ann_r2_mean"]

    ax.axvline(best_lambda, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.scatter([best_lambda], [best_r2], color="red", s=180, zorder=5, marker="*")
    # 标注避开数据点重叠区域
    if best_lambda < 1.0:
        xytext = (best_lambda * 5, best_r2 + 0.04)
    else:
        xytext = (best_lambda * 0.3, best_r2 + 0.04)
    ax.annotate(f"\u03bb = {best_lambda}\nR\u00b2 = {best_r2:.3f}",
                xy=(best_lambda, best_r2),
                xytext=xytext,
                fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("\u03bb（KL散度权重）", fontsize=14)
    ax.set_ylabel("R\u00b2", fontsize=14)
    ax.set_title("（a）KL散度权重消融", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # ----- Panel (b): 样本量敏感性 -----
    ax = axes[1]

    for model, color, marker, label in [("RF", "#e53935", "o", "随机森林"), ("LGBM", "#fb8c00", "s", "LightGBM")]:
        data = df_lc[df_lc["model"] == model].sort_values("n_train")
        ax.plot(data["n_train"], data["r2_mean"], f"{marker}-",
                color=color, label=label, linewidth=2.5, markersize=8)
        ax.fill_between(data["n_train"],
                        data["r2_mean"] - data["r2_std"],
                        data["r2_mean"] + data["r2_std"],
                        alpha=0.2, color=color)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("训练样本量（n）", fontsize=14)
    ax.set_ylabel("R\u00b2", fontsize=14)
    ax.set_title("（b）样本量敏感性", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    save_figure(fig, "fig6_ablation_zh")
    print("  Fig 6 完成")


# ============================================================================
# Fig 7: PCL-AE潜空间对比 - 中文版
# ============================================================================
def make_fig7_latent_space():
    """生成潜空间对比图：全中文"""

    df_latent = pd.read_csv(PATH_LATENT, encoding="utf-8-sig")

    models = df_latent["model"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ----- Panel (a): 特征相关维度数量 -----
    ax = axes[0]
    x = np.arange(len(models))
    n_dims_r = df_latent["n_dims_with_|r|>0.1"].values
    n_dims_total = df_latent["n_dims"].values

    bars = ax.bar(x, n_dims_r, color=["#1976D2", "#e53935"], edgecolor="black", width=0.5, linewidth=1.0)
    ax.bar(x, n_dims_total - n_dims_r, bottom=n_dims_r, color=["#90CAF9", "#FFCDD2"], edgecolor="black", width=0.5, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(["标准AE", "PCL-AE"], fontsize=13)
    ax.set_ylabel("维度数量", fontsize=14)
    ax.set_title("（a）特征相关维度数量", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, n_r, n_total in zip(bars, n_dims_r, n_dims_total):
        ax.text(bar.get_x() + bar.get_width()/2, n_r/2,
                f"{int(n_r)}/{int(n_total)}", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

    ax.legend([mpatches.Patch(color="#90CAF9"), mpatches.Patch(color="#1976D2")],
              ["不相关", "相关（|r|>0.1）"], loc="upper left", fontsize=11)

    # ----- Panel (b): 质量指标 -----
    ax = axes[1]

    fractions = df_latent["fraction_x"].values
    mean_rs = df_latent["mean_|r|"].values

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, fractions, width, label="fraction_x", color="#1976D2", edgecolor="black", linewidth=1.0)
    ax.bar(x + width/2, mean_rs, width, label="mean_|r|", color="#2ca02c", edgecolor="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(["标准AE", "PCL-AE"], fontsize=13)
    ax.set_ylabel("得分", fontsize=14)
    ax.set_title("（b）潜空间质量", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for i, (f, r) in enumerate(zip(fractions, mean_rs)):
        ax.text(i - width/2, f + 0.04, f"{f:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.text(i + width/2, r + 0.04, f"{r:.3f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.0)

    # ----- Panel (c): 产率相关性与聚类 -----
    ax = axes[2]

    rho_yield = df_latent["rho_yield"].values
    silhouette = df_latent["silhouette_catalyst_family"].values

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, rho_yield, width, label="\u03c1(产率)", color="#ff9800", edgecolor="black", linewidth=1.0)
    ax.bar(x + width/2, silhouette, width, label="轮廓系数", color="#9c27b0", edgecolor="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(["标准AE", "PCL-AE"], fontsize=13)
    ax.set_ylabel("系数", fontsize=14)
    ax.set_title("（c）产率相关性与聚类", fontweight="bold", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    for i, (r, s) in enumerate(zip(rho_yield, silhouette)):
        ax.text(i - width/2, r + 0.03, f"{r:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.text(i + width/2, s + 0.03, f"{s:.3f}", ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("PCL-AE vs 标准AE 潜空间对比", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig7_latent_space_zh")
    print("  Fig 7 完成")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """生成所有8张论文图表（中文版）"""

    print("=" * 70)
    print("Generating Paper Figures v6 (Chinese)")
    print("=" * 70)
    print(f"Output: {FIGURES_DIR}")
    if MUPDF_AVAILABLE:
        print("SVG conversion: enabled (PyMuPDF)")
    else:
        print("SVG conversion: PyMuPDF not available (pip install pymupdf)")
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
    print(f"Done! Figures generated in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
