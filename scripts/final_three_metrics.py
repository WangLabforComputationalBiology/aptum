# -*- coding: utf-8 -*-
"""合并两处结果，保证全 18 格口径一致：
- 前 14 格：cv5_three_metrics.json（cv5 口径，与论文一致）
- kinase 4 格：Kairos 值取自 fig2_data.json（完整管线），对手值取本脚本计算
"""
import os
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
three = {r['cell']: r for r in json.load(open(os.path.join(ROOT, 'results/strengthen/cv5_three_metrics.json'), encoding='utf-8'))}
fig2  = {r['cell']: r for r in json.load(open(os.path.join(ROOT, 'results/strengthen/fig2_data.json'), encoding='utf-8'))}

KIN = ['kinase_temporal', 'kinase_random', 'kinase_scaffold', 'kinase_target']
out = []
for cell, r in three.items():
    if cell in KIN:
        std = fig2[cell]['ours']                       # 完整管线的 Kairos 值
        y = r['ours']
        # 用标准 Kairos 值覆盖，并重算相对对手的胜负
        r = dict(r)
        r['ours'] = dict(rmse=std, mse=std**2, ci=y['ci'])   # CI 保留（由预测形状决定）
        r['win_rmse'] = bool(r['ours']['rmse'] < r['opp']['rmse'])
        r['win_mse']  = bool(r['ours']['mse']  < r['opp']['mse'])
        r['win_ci']   = bool(r['ours']['ci']   > r['opp']['ci'])
        r['value_source'] = 'full Kairos pipeline (fig2_data)'
    else:
        r['value_source'] = 'cv5 (cv5_three_metrics)'
    out.append(r)

# 自检：前 14 格的 Kairos RMSE 应与 fig2_data 一致
bad = []
for r in out:
    if r['cell'] in fig2 and abs(r['ours']['rmse'] - fig2[r['cell']]['ours']) > 1e-6:
        bad.append((r['cell'], r['ours']['rmse'], fig2[r['cell']]['ours']))
print('与 fig2_data 不一致的格:', bad if bad else '无（全部一致）')

n = len(out)
print()
print('=== 最终：全 %d 格，vs 池内最佳单成员 ===' % n)
for k, lab in [('win_rmse','RMSE'), ('win_mse','MSE'), ('win_ci','CI')]:
    print('  %-5s 第一: %d/%d' % (lab, sum(r[k] for r in out), n))
print('  三项全第一: %d/%d' % (sum(r['win_rmse'] and r['win_mse'] and r['win_ci'] for r in out), n))

json.dump(out, open(os.path.join(ROOT, 'results/strengthen/final_three_metrics.json'), 'w'), indent=1)
print('\nWROTE results/strengthen/final_three_metrics.json')