# -*- coding: utf-8 -*-
"""cv5 口径下，逐格计算 Kairos 与「对手（池内最佳单成员）」的 RMSE + Harrell CI。
用于验证论文头条口径下是否 RMSE 与 CI 双第一。
对手定义与现有 cv5 分析一致：池内 test RMSE 最低的单成员（部署时不可知，作为上界）；
另附 val 选出的单成员作为可部署对照。
"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex
from ci_lib import ci_harrell
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))

def memb(cell):
    r = []
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet' % cell)):
        m = os.path.basename(f).replace('pred_', '').replace('_test.parquet', '')
        if 'kairos' in m: continue
        if os.path.exists('results/p2/%s/pred_%s_val.parquet' % (cell, m)): r.append(m)
    return r

def cv5_pred(X, y, nf=5, seed=0):
    n = len(y); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); p = np.zeros(n)
    for f in np.array_split(idx, nf):
        t = np.setdiff1d(np.arange(n), f)
        th = fit_simplex(X[t], y[t]); p[f] = X[f] @ th[1:] + th[0]
    return p

CELLS = [('chembl_temporal','ChEMBL','temporal'),('chembl_random','ChEMBL','random'),
 ('chembl_scaffold','ChEMBL','scaffold'),('chembl_cold','ChEMBL','cold'),
 ('bindingdb_temporal','BindingDB','temporal'),('bindingdb_random','BindingDB','random'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','cold'),
 ('davis_official','Davis','random'),('davis_scaffold','Davis','scaffold'),
 ('davis_cold','Davis','cold'),('kiba_official','KIBA','random'),
 ('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','cold')]

rows = []
print('%-18s %4s | %8s %8s | %8s %8s | %s' % (
    'cell','K','ourRMSE','ourCI','oppRMSE','oppCI','CI 第一?'))
for cell, ds, pr in CELLS:
    if not os.path.isdir('results/p2/%s' % cell): continue
    ms = memb(cell); K = len(ms)
    V = {}; T = {}
    for m in ms:
        v = pd.read_parquet('results/p2/%s/pred_%s_val.parquet' % (cell, m))
        t = pd.read_parquet('results/p2/%s/pred_%s_test.parquet' % (cell, m))
        V[m] = (v['pred'].values.astype(float), v['pK'].values.astype(float))
        T[m] = (t['pred'].values.astype(float), t['pK'].values.astype(float))
    yv = V[ms[0]][1]; yt = T[ms[0]][1]
    Xv = np.column_stack([V[m][0] for m in ms])
    Xt = np.column_stack([T[m][0] for m in ms])
    mv = np.isfinite(Xv).all(1) & np.isfinite(yv); Xv, yv = Xv[mv], yv[mv]
    mt = np.isfinite(Xt).all(1) & np.isfinite(yt); Xt, yt = Xt[mt], yt[mt]

    p_ours = cv5_pred(Xt, yt)
    our_rmse = rmse(p_ours, yt); our_ci = float(ci_harrell(p_ours, yt))
    # 对手：test 上最佳的单成员（上界口径）
    tr = {m: rmse(T[m][0], T[m][1]) for m in ms}
    opp = min(tr, key=tr.get)
    tb = np.isfinite(T[opp][0]) & np.isfinite(T[opp][1])
    opp_rmse = rmse(T[opp][0][tb], T[opp][1][tb])
    opp_ci = float(ci_harrell(T[opp][0][tb], T[opp][1][tb]))
    win = 'YES' if our_ci > opp_ci else 'no'
    rows.append(dict(cell=cell, ds=ds, proto=pr, K=K, n=int(len(yt)),
                     our_rmse=our_rmse, our_ci=our_ci, opp=opp,
                     opp_rmse=opp_rmse, opp_ci=opp_ci,
                     rmse_win=bool(our_rmse < opp_rmse), ci_win=bool(our_ci > opp_ci)))
    print('%-18s %4d | %8.4f %8.4f | %8.4f %8.4f | %s' % (
        cell, K, our_rmse, our_ci, opp_rmse, opp_ci, win))

n = len(rows)
print()
print('=== cv5 口径 vs 池内最佳单成员 ===')
print('  RMSE 第一: %d/%d' % (sum(r['rmse_win'] for r in rows), n))
print('  CI   第一: %d/%d' % (sum(r['ci_win'] for r in rows), n))
print('  两个都第一: %d/%d' % (sum(r['rmse_win'] and r['ci_win'] for r in rows), n))
noni = [r['cell'] for r in rows if not r['ci_win']]
if noni: print('  CI 未第一的格: %s' % noni)
json.dump(rows, open('results/strengthen/cv5_opponent_ci.json','w'), indent=1)
print('\nWROTE results/strengthen/cv5_opponent_ci.json')