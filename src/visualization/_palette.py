# -*- coding: utf-8 -*-
"""
_palette.py — 论文全图统一调色板（单一颜色真相来源）
与 fig0_graphical_abstract_v3.py 颜色体系完全对齐。

Usage:
    from _palette import SUB, MODEL, NEUTRAL, setup_chinese_font
"""

import os
import matplotlib
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# ── CJK 中文字体自动设置 ─────────────────────────────────────────────────────
def setup_chinese_font():
    """自动检测并设置中文字体，返回 font family 列表供 rcParams 使用。"""
    avail = [f.name for f in fm.fontManager.ttflist]
    cjk_kws = ["simhei", "yahei", "wenquanyi", "noto sans cjk",
               "noto sans sc", "source han", "arial unicode ms",
               "noto serif cjk"]
    cjk = [f for f in avail if any(k in f.lower() for k in cjk_kws)]
    if cjk:
        matplotlib.rcParams["font.sans-serif"] = [cjk[0]] + matplotlib.rcParams["font.sans-serif"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                                   "Arial Unicode MS",
                                                   "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

# ── 5 底物颜色 ────────────────────────────────────────────────────────────────
# 与 fig0_graphical_abstract_v3.py 完全一致（v3 Q1 期刊配色）
SUB = {
    "CHO":  "#E63946",   # 鲜明红 — 重点底物，每图必标
    "PO":   "#457B9D",   # 深蓝
    "SO":   "#1D3557",   # 藏蓝
    "ECH":  "#2A9D8F",   # 青绿
    "IGE":  "#E09F3E",   # 金橙
}
SUB_LABEL = {
    "CHO":  "CHO\n(cyclohexene oxide)",
    "PO":   "PO\n(propylene oxide)",
    "SO":   "SO\n(styrene oxide)",
    "ECH":  "ECH\n(epichlorohydrin)",
    "IGE":  "IGE\n(isopropyl glycidyl ether)",
}
SUB_ZH = {
    "CHO":  "CHO（环己烯氧化物）",
    "PO":   "PO（环氧丙烷）",
    "SO":   "SO（苯乙烯氧化物）",
    "ECH":  "ECH（表氯醇）",
    "IGE":  "IGE（异丙基缩水甘油醚）",
}

# ── 4 模型颜色 ────────────────────────────────────────────────────────────────
MODEL = {
    "XGB":           "#3b5a82",   # 深蓝灰
    "LGBM":          "#5a7a9a",   # 中蓝灰
    "RF":            "#7d96ae",   # 浅蓝灰
    "DualBranchANN": "#9c3024",   # 强调红
}

# ── 中性色 ────────────────────────────────────────────────────────────────────
NEUTRAL = {
    "primary":   "#1D3557",   # 主文字
    "secondary": "#444444",   # 次文字
    "tertiary":  "#888888",   # 辅助文字
    "grid":      "#CCCCCC",   # 网格线
    "zero":      "#222222",   # 零基线
    "highlight": "#E63946",   # 高亮（同 CHO）
}

# ── 催化剂类型颜色（热图用）────────────────────────────────────────────────────
CAT_PALETTE = [
    "#4A90D9",   # 离子液体 IL
    "#50B432",   # 金属卤化物
    "#ED561B",   # 混合体系
    "#FF9500",   # 有机碱
    "#AAAAAA",   # 未知
]
CAT_LABEL_ZH = ["离子液体", "金属卤化物", "混合体系", "有机碱", "未知"]
CAT_LABEL_EN = ["Ionic liquid", "Metal halide", "Mixed system", "Organic base", "Unknown"]

# ── Q1 期刊美学辅助 ───────────────────────────────────────────────────────────
# 标准：去饱和、颜色盲友好、学术风格
Q1_DEEP  = "#1D355A"   # 深藏青（主标题）
Q1_MID   = "#457B9D"   # 中蓝（次级标注）
Q1_LIGHT = "#A8DADC"   # 浅青（填充）
Q1_ACCENT= "#E63946"   # 强调红（CHO 高亮）
Q1_GRAY  = "#6C757D"   # 中灰（次级文字）

if __name__ == "__main__":
    # 快速调色板预览
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, (name, color) in enumerate(SUB.items()):
        ax.scatter(i, 0, s=800, color=color, edgecolor="white", linewidth=2)
        ax.text(i, 0.18, name, ha="center", fontsize=11, fontweight="bold")
        ax.text(i, -0.18, SUB_ZH[name], ha="center", fontsize=8, color="#666")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.4, 0.4)
    ax.axis("off")
    ax.set_title("SUB palette — Q1 journal colors", fontsize=12, pad=10)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/_palette_preview.png", dpi=150, bbox_inches="tight")
    print("Saved figures/_palette_preview.png")
