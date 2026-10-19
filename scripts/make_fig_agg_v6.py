# -*- coding: utf-8 -*-
"""Figure 3 — 18-cell evaluation, three metrics（matplotlib 矢量，与 fig_eds 同风格）
a: RMSE 哑铃（uniform average / best single member / Kairos）
b: 相对降幅（vs 两个基线）—— 条间距拉开、数值只标外侧
c: 三指标优势（RMSE / MSE / CI），对池内最佳单成员
"""
import os
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D   = json.load(open(os.path.join(ROOT, 'results/strengthen/fig2_data.json'), encoding='utf-8'))
T3  = json.load(open(os.path.join(ROOT, 'results/strengthen/final_three_metrics.json'), encoding='utf-8'))
OUT = os.path.join(ROOT, 'figs', 'fig_aggregation.pdf')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
})
DARK, MID, LIGHT = '#2B466F', '#3F6399', '#A8BDD6'
GREY = '#9AA5B1'

rows = D
n = len(rows)
y = np.arange(n)[::-1]

fig = plt.figure(figsize=(7.15, 3.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 0.94, 0.86], wspace=0.10,
                      left=0.150, right=0.985, top=0.895, bottom=0.155)

# ==================== (a) dumbbell ====================
ax = fig.add_subplot(gs[0])
dy = 0.24                      # 每行内三个标记的垂直错位量
for yi, r in zip(y, rows):
    ax.plot([min(r['uni'], r['bs'], r['ours']), max(r['uni'], r['bs'], r['ours'])],
            [yi, yi], color='#DCE4EF', lw=0.9, zorder=0)
    ax.plot(r['uni'], yi + dy, marker='D', ms=3.4, color=GREY, zorder=2)
    ax.plot(r['bs'], yi, marker='s', ms=3.4, mfc='none', mec=GREY, mew=0.9, zorder=2)
    ax.plot(r['ours'], yi - dy, marker='o', ms=4.4, color=MID, zorder=3)
    for xv, yo, col in [(r['uni'], yi + dy, GREY), (r['bs'], yi, GREY),
                        (r['ours'], yi - dy, MID)]:
        ax.plot([xv - 0.012, xv + 0.012], [yo, yo], color=col, lw=1.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['ds']} · {r['proto']}" for r in rows])
ax.set_xlabel('RMSE (lower is better)')
ax.set_title('(a) Absolute RMSE', pad=6)
ax.set_xlim(0.38, 1.34)
ax.set_ylim(-0.85, len(rows) - 0.15)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', length=2)
ax.grid(axis='x', color='#E4EAF3', lw=0.6); ax.set_axisbelow(True)
ax.legend(handles=[
    plt.Line2D([], [], marker='D', ls='', ms=3.4, color=GREY, label='uniform avg.'),
    plt.Line2D([], [], marker='s', ls='', ms=3.4, mfc='none', mec=GREY, mew=0.9,
               label='best single member'),
    plt.Line2D([], [], marker='o', ls='', ms=4.4, color=MID, label='Kairos'),
], loc='lower center', bbox_to_anchor=(0.5, -0.215), ncol=3, frameon=False,
   handletextpad=0.30, borderpad=0.0, columnspacing=1.1)
prev = None
for yi, r in zip(y, rows):
    if r['ds'] != prev:
        if prev is not None: ax.axhline(yi + 0.5, color='#E4EAF3', lw=0.7)
        prev = r['ds']

# ==================== (b) reduction over two baselines ====================
ax2 = fig.add_subplot(gs[1])
h = 0.30
for yi, r in zip(y, rows):
    ax2.barh(yi + h/2 + 0.045, r['gain_uni'], height=h, color=MID, edgecolor='none')
    ax2.barh(yi - h/2 - 0.045, r['gain_bs'], height=h, color=DARK, edgecolor='none')
for yi, r in zip(y, rows):
    # 数值只标外侧，且较长者在外，避免相撞
    gu, gb = r['gain_uni'], r['gain_bs']
    ax2.text(gu + 0.35, yi + h/2 + 0.045, f'{gu:.1f}', va='center',
             fontsize=6.2, color=MID)
    ax2.text(gb + 0.35, yi - h/2 - 0.045, f'{gb:.1f}', va='center',
             fontsize=6.2, color=DARK)
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel('reduction (%)')
ax2.set_title('(b) Reduction over baselines', pad=6)
ax2.set_xlim(0, 27)
ax2.spines[['top', 'right', 'left']].set_visible(False)
ax2.tick_params(axis='y', length=0); ax2.tick_params(axis='x', length=2)
ax2.grid(axis='x', color='#E4EAF3', lw=0.6); ax2.set_axisbelow(True)
ax2.legend(handles=[
    plt.Line2D([], [], marker='s', ls='', ms=4.2, color=MID, label='vs. uniform avg.'),
    plt.Line2D([], [], marker='s', ls='', ms=4.2, color=DARK, label='vs. best single'),
], loc='lower right', frameon=False, handletextpad=0.35, borderpad=0.15,
   labelspacing=0.30)

# ==================== (c) three-metric advantage ====================
ax3 = fig.add_subplot(gs[2])
t3 = {r['cell']: r for r in T3}
key = {'chembl_temporal':'chembl_temporal','chembl_random':'chembl_random',
       'chembl_scaffold':'chembl_scaffold','chembl_cold':'chembl_cold',
       'bindingdb_temporal':'bindingdb_temporal','bindingdb_random':'bindingdb_random',
       'bindingdb_scaffold':'bindingdb_scaffold','bindingdb_cold':'bindingdb_cold',
       'davis_official':'davis_official','davis_scaffold':'davis_scaffold',
       'davis_cold':'davis_cold','kiba_official':'kiba_official',
       'kiba_scaffold':'kiba_scaffold','kiba_cold':'kiba_cold',
       'kinase_temporal':'kinase_temporal','kinase_random':'kinase_random',
       'kinase_scaffold':'kinase_scaffold','kinase_target':'kinase_target'}
adv_r, adv_m, adv_c = [], [], []
for r in rows:
    c = r['cell']; t = t3[c]
    adv_r.append(100*(t['opp']['rmse'] - t['ours']['rmse']) / t['opp']['rmse'])
    adv_m.append(100*(t['opp']['mse']  - t['ours']['mse'])  / t['opp']['mse'])
    adv_c.append(100*(t['ours']['ci']  - t['opp']['ci']))
metrics = [('RMSE', adv_r, MID), ('MSE', adv_m, DARK), ('CI (pp)', adv_c, LIGHT)]
bw = 0.26
xs = np.arange(3)
for i, (lab, v, col) in enumerate(metrics):
    bp = ax3.boxplot([v], positions=[i], widths=bw*2.6, patch_artist=True,
                     medianprops=dict(color='white', lw=1.0),
                     boxprops=dict(facecolor=col, edgecolor='none'),
                     whiskerprops=dict(color=col, lw=0.7),
                     capprops=dict(color=col, lw=0.7),
                     flierprops=dict(marker='o', ms=2.2, mfc=col, mec='none'))
    ax3.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)),
                v, s=5, color='#333333', alpha=0.55, zorder=3, lw=0)
    # mean 标注放在箱体下方，避免压到标题
    ax3.text(i, -1.55, f'{np.mean(v):.1f}', ha='center', fontsize=7.0,
             color=col, fontweight='bold')
ax3.set_xticks(xs)
ax3.set_xticklabels([m[0] for m in metrics])
ax3.set_ylabel('advantage over best single member\n(% for RMSE/MSE, points for CI)')
ax3.set_title('(c) Three-metric advantage', pad=6)
ax3.axhline(0, color='#999', lw=0.7)
ax3.set_ylim(-2.8, 17)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(length=2)
ax3.grid(axis='y', color='#E4EAF3', lw=0.6); ax3.set_axisbelow(True)

fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02)
print('WROTE', OUT)
print('RMSE adv mean %+.2f%%  MSE %+.2f%%  CI %+.2fpp' % (
    np.mean(adv_r), np.mean(adv_m), np.mean(adv_c)))
print('all positive:', all(x > 0 for x in adv_r), all(x > 0 for x in adv_m), all(x > 0 for x in adv_c))