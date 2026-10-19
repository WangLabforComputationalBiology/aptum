# -*- coding: utf-8 -*-
"""Figure 2 (fig:eds) — (C0) endpoint-divergence selection
a: 秩热图（9 成员 × 3 端点，1 = best）——δ 的来源
b: δ_k 条形图 + 池中位阈值线——C0 的决策过程
c: 选择稳定性（50 次自助重采样）
风格与其余图一致（同一 rcParams）。
"""
import os
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = json.load(open(os.path.join(ROOT, 'results/strengthen/eds_component.json'), encoding='utf-8'))
R = json.load(open(os.path.join(ROOT, 'results/strengthen/eds_robust.json'), encoding='utf-8'))
OUT = os.path.join(ROOT, 'figs', 'fig_eds.pdf')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
})
DARK, MID, LIGHT = '#2B466F', '#3F6399', '#A8BDD6'
GREY, ACC = '#9AA5B1', '#B4574A'

eps = E['rank_endpoints']
ranks = E['ranks']; delta = E['delta']; thr = E['threshold']; Asel = set(E['A_eds'])
mems = sorted(ranks, key=lambda m: -delta[m])

def pretty(m):
    return r'$f_1$' if m == 'f1' else m

def pretty_ep(e):
    return e.replace('IC50', r'IC$_{50}$').replace('Kd', r'$K_d$').replace('Ki', r'$K_i$')

fig = plt.figure(figsize=(7.1, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.00, 0.92], wspace=0.46,
                      left=0.085, right=0.985, top=0.855, bottom=0.165)

# ==================== (a) rank heatmap ====================
ax = fig.add_subplot(gs[0])
M = np.array([ranks[m] for m in mems], dtype=float)
im = ax.imshow(1.0 / M, cmap='Blues', vmin=0.11, vmax=1.0, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        r = M[i, j]
        ax.text(j, i, f'{int(r)}', ha='center', va='center', fontsize=7,
                color='white' if r <= 4 else '#333333')
ax.set_xticks(range(len(eps)))
ax.set_xticklabels([pretty_ep(e) for e in eps])
ax.set_yticks(range(len(mems)))
ax.set_yticklabels([pretty(m) for m in mems])
for i, m in enumerate(mems):
    if m in Asel:
        ax.get_yticklabels()[i].set_fontweight('bold')
ax.set_title('(a) Endpoint rank (1 = best)', pad=6)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('1/rank', fontsize=6.5, labelpad=1)
cb.ax.tick_params(labelsize=6, length=1.2, pad=1)
cb.outline.set_linewidth(0.5)

# ==================== (b) delta bar + threshold ====================
ax2 = fig.add_subplot(gs[1])
xs = np.arange(len(mems))[::-1]
vas = [delta[m] for m in mems]
cols = [MID if m in Asel else LIGHT for m in mems]
ax2.barh(xs, vas, color=cols, height=0.66, edgecolor='none')
ax2.axvline(thr, color=ACC, lw=1.0, ls='--', zorder=3)
ax2.text(thr + 0.05, -0.85, f'pool median {thr:.2f}', fontsize=6.6,
         color=ACC, va='center')
for x, v, m in zip(xs, vas, mems):
    ax2.text(v + 0.045, x, f'{v:.2f}', va='center', fontsize=6.6,
             color=MID if m in Asel else '#888888')
ax2.set_yticks(xs)
ax2.set_yticklabels([pretty(m) for m in mems])
for i, m in enumerate(mems):
    if m in Asel:
        ax2.get_yticklabels()[i].set_fontweight('bold')
ax2.set_xlim(0, 2.95)
ax2.set_ylim(-1.5, len(mems) - 0.3)
ax2.set_xlabel('endpoint divergence $\\delta_k$')
ax2.set_title('(b) Divergence and selection', pad=6)
ax2.spines[['top', 'right', 'left']].set_visible(False)
ax2.tick_params(axis='y', length=0); ax2.tick_params(axis='x', length=2)
ax2.grid(axis='x', color='#E4EAF3', lw=0.6); ax2.set_axisbelow(True)

# ==================== (c) bootstrap stability ====================
ax3 = fig.add_subplot(gs[2])
picks = R['bootstrap_pick_freq']
pm = sorted(picks, key=lambda m: -picks[m])
pv = [picks[m] for m in pm]
yc = np.arange(len(pm))[::-1]
cols3 = [MID if p >= 45 else (LIGHT if p >= 20 else '#E8EDF4') for p in pv]
ax3.barh(yc, pv, color=cols3, height=0.66, edgecolor='none')
for yi, p in zip(yc, pv):
    ax3.text(p + 1.0, yi, str(p), va='center', fontsize=6.6)
ax3.set_yticks(yc)
ax3.set_yticklabels([pretty(m) for m in pm])
for i, m in enumerate(pm):
    if picks[m] >= 45:
        ax3.get_yticklabels()[i].set_fontweight('bold')
ax3.set_xlim(0, 60)
ax3.set_xlabel('times selected (of 50)')
ax3.set_title('(c) Bootstrap stability', pad=6)
ax3.spines[['top', 'right', 'left']].set_visible(False)
ax3.tick_params(axis='y', length=0); ax3.tick_params(axis='x', length=2)
ax3.grid(axis='x', color='#E4EAF3', lw=0.6); ax3.set_axisbelow(True)

fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02)
print('WROTE', OUT)
print('selected (above median):', [m for m in mems if m in Asel])
print('solid picks (>=45/50):', [m for m in pm if picks[m] >= 45])