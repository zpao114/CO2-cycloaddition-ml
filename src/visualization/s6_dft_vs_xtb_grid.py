# -*- coding: utf-8 -*-
"""
figures/s6_dft_vs_xtb_grid.py
=============================
One-shot figure regeneration: 4×4 scatter matrix of DFT vs xTB descriptors
from `dft_validation/514_dft_vs_xtb_report.csv`.

Output: figures/s6_dft_vs_xtb_grid.png/pdf
        figures/s6_dft_vs_xtb_homo.png/pdf   (单独 HOMO — 论文主图)
"""
import os, sys, io, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

# CJK font (reuse logic from regen_all_v3)
import matplotlib.font_manager as fm
avail = [f.name for f in fm.fontManager.ttflist]
cjk = ['simhei', 'wenquanyi', 'yahei', 'noto sans cjk', 'noto sans sc',
       'microsoft yahei', 'source han', 'arial unicode ms']
cjk_fonts = [f for f in avail if any(k in f.lower() for k in cjk)]
if cjk_fonts:
    plt.rcParams['font.sans-serif'] = [cjk_fonts[0]] + plt.rcParams['font.sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROOT = r'D:\machine-learning\CO2-cycloaddition'
os.chdir(ROOT)

SRC = 'dft_validation/results/514_dft_vs_xtb_report.csv'
df = pd.read_csv(SRC)
print(f'Loaded {len(df)} molecules from {SRC}')

# Subset flag — TBAI ions flagged for exclusion (matches SI §S6.2)
def is_tbai(s):
    return isinstance(s, str) and 'TBAI' in s
df['is_tbai'] = df['name'].apply(is_tbai)
df['is_terminal_epoxide'] = df['name'].isin([
    'styrene_oxide', 'epichlorohydrin', 'propylene_oxide', 'epoxybutane',
    'isopropyl_glycidyl_ether', 'allyl_glycidyl_ether',
    'furfuryl_glycidyl_ether', 'phenyl_glycidyl_ether',
])
df['is_cho'] = df['name'] == 'cyclohexene_oxide'

# Descriptors to plot
descriptors = [
    ('homo_eV',  'HOMO (eV)',  'HOMO'),
    ('lumo_eV',  'LUMO (eV)',  'LUMO'),
    ('gap_eV',   'Gap (eV)',   'HOMO–LUMO Gap'),
    ('dipole_D', 'Dipole (D)', 'Dipole'),
]

# Pre-compute clean subsets (drop TBAI ions, per SI §S6.2)
clean = df[~df['is_tbai']].copy()
print(f'Clean subset (N={len(clean)}) drops {df["is_tbai"].sum()} TBAI ions')

# ---- Figure 1: 2×2 grid, all 4 descriptors ---------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for ax, (col, label, title) in zip(axes, descriptors):
    xtb_col, dft_col = f'xtb_{col}', f'dft_{col}'

    # Plot non-TBAI molecules first (in background)
    non_tbai = df[~df['is_tbai']]
    ax.scatter(non_tbai[xtb_col], non_tbai[dft_col],
               s=60, c='#cccccc', edgecolor='gray', alpha=0.7, zorder=1,
               label=f'non-substrate (N={len(non_tbai)})')

    # Highlight terminal epoxides
    term = df[df['is_terminal_epoxide']]
    ax.scatter(term[xtb_col], term[dft_col],
               s=120, c='#d62728', edgecolor='black', linewidth=0.8,
               alpha=0.9, zorder=3,
               label=f'terminal epoxide (n={len(term)})')

    # Highlight CHO
    cho = df[df['is_cho']]
    if len(cho) > 0:
        ax.scatter(cho[xtb_col], cho[dft_col],
                   s=160, c='#1f77b4', edgecolor='black', linewidth=1.0,
                   alpha=1.0, zorder=4, marker='s',
                   label='cyclohexene oxide (CHO)')

    # Plot TBAI ions (excluded, marked with ×)
    tbai = df[df['is_tbai']]
    ax.scatter(tbai[xtb_col], tbai[dft_col],
               s=100, c='#888888', alpha=0.5, zorder=2,
               marker='x', linewidths=2,
               label=f'TBAI ion (excluded, n={len(tbai)})')

    # y = x reference
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=0)

    # Annotate Spearman on clean subset (skip if column has no valid data)
    valid_col = clean[[xtb_col, dft_col]].dropna()
    if len(valid_col) >= 3:
        rho, _ = spearmanr(valid_col[xtb_col], valid_col[dft_col])
        r, _ = pearsonr(valid_col[xtb_col], valid_col[dft_col])
        ax.text(0.05, 0.95,
                f'Spearman ρ = {rho:+.3f}\nPearson R = {r:+.3f}\n(N={len(valid_col)}, TBAI excl.)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.85))

    ax.set_xlabel(f'xTB {label}', fontsize=11)
    ax.set_ylabel(f'DFT {label}', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25)

# Single legend (collect handles from first axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('DFT (B3LYP-D3BJ/def2-TZVP) vs GFN2-xTB descriptors\n'
             '15-molecule calibration set (3 TBAI ions excluded)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

out1 = 'figures/s6_dft_vs_xtb_grid'
fig.savefig(out1 + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out1 + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Wrote {out1}.{{png,pdf}}')

# ---- Figure 2: HOMO-only — single panel, main paper figure -----------------
fig, ax = plt.subplots(figsize=(7, 6))

xtb_col, dft_col = 'xtb_homo_eV', 'dft_homo_eV'
non_tbai = df[~df['is_tbai']]
ax.scatter(non_tbai[xtb_col], non_tbai[dft_col],
           s=80, c='#cccccc', edgecolor='gray', alpha=0.7, zorder=1,
           label=f'other (N={len(non_tbai)})')

term = df[df['is_terminal_epoxide']]
ax.scatter(term[xtb_col], term[dft_col],
           s=180, c='#d62728', edgecolor='black', linewidth=0.8,
           alpha=0.95, zorder=3,
           label=f'terminal epoxide (n={len(term)})')

cho = df[df['is_cho']]
ax.scatter(cho[xtb_col], cho[dft_col],
           s=200, c='#1f77b4', edgecolor='black', linewidth=1.2,
           alpha=1.0, zorder=4, marker='s',
           label='cyclohexene oxide')

tbai = df[df['is_tbai']]
ax.scatter(tbai[xtb_col], tbai[dft_col],
           s=120, c='#888888', alpha=0.6, zorder=2,
           marker='x', linewidths=2,
           label=f'TBAI ion (excluded, n={len(tbai)})')

# Annotate each terminal epoxide with its name
for _, r in term.iterrows():
    ax.annotate(r['name'].replace('_', '\n'),
                (r[xtb_col], r[dft_col]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7.5, alpha=0.7)

# y = x line
lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=0)

# Spearman stats
rho_full, _ = spearmanr(df[~df['is_tbai']][xtb_col], df[~df['is_tbai']][dft_col])
rho_term, _ = spearmanr(term[xtb_col], term[dft_col])

ax.text(0.05, 0.95,
        f'All 15:   Spearman ρ = {rho_full:+.3f}\nTerminal 8: Spearman ρ = {rho_term:+.3f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.9))

ax.set_xlabel('xTB GFN2 HOMO (eV)', fontsize=12)
ax.set_ylabel('DFT B3LYP-D3BJ/def2-TZVP HOMO (eV)', fontsize=12)
ax.set_title('HOMO ranking: xTB vs DFT — terminal epoxides show perfect rank-order agreement',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
out2 = 'figures/s6_dft_vs_xtb_homo'
fig.savefig(out2 + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out2 + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Wrote {out2}.{{png,pdf}}')

print('\n--- Summary on clean (N=15) subset ---')
for col, label, title in descriptors:
    xtb_c, dft_c = f'xtb_{col}', f'dft_{col}'
    valid = clean[[xtb_c, dft_c]].dropna()
    if len(valid) < 3:
        print(f'  {title:18s}  SKIPPED (insufficient non-NaN pairs)')
        continue
    rho, _ = spearmanr(valid[xtb_c], valid[dft_c])
    r, _ = pearsonr(valid[xtb_c], valid[dft_c])
    mae = (valid[xtb_c] - valid[dft_c]).abs().mean()
    print(f'  {title:18s}  Spearman ρ = {rho:+.4f}   Pearson R = {r:+.4f}   MAE = {mae:.3f}')