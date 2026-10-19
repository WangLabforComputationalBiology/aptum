# -*- coding: utf-8 -*-
"""cv5 口径下，全 18 格（含 kinase 4 格）的 Kairos 与对手在 RMSE / MSE / CI 三指标上的对比。
对手 = 池内 test RMSE 最低的单成员（不可部署的口径，作为上界）。
"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex
from ci_lib import ci_harrell
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rmse(p, y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))

def cv5_pred(X, y, nf=5, seed=0):
    n = len(y); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); p = np.zeros(n)
    for f in np.array_split(idx, nf):
        t = np.setdiff1d(np.arange(n), f)
        th = fit_simplex(X[t], y[t]); p[f] = X[f] @ th[1:] + th[0]
    return p

def avail(prefix, cell):
    """返回该格可用于 cv5 的成员名列表（test 与 val 都有预测）。"""
    tset = {os.path.basename(f)[5:].replace('_test.parquet', '')
            for f in glob.glob('%s/%s/pred_*_test.parquet' % (prefix, cell))}
    vset = {os.path.basename(f)[5:].replace('_val.parquet', '')
            for f in glob.glob('%s/%s/pred_*_val.parquet' % (prefix, cell))}
    return sorted(m for m in tset & vset if m and 'kairos' not in m)

CELLS = [('chembl_temporal','ChEMBL','temporal'),('chembl_random','ChEMBL','random'),
 ('chembl_scaffold','ChEMBL','scaffold'),('chembl_cold','ChEMBL','cold'),
 ('bindingdb_temporal','BindingDB','temporal'),('bindingdb_random','BindingDB','random'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','cold'),
 ('davis_official','Davis','random'),('davis_scaffold','Davis','scaffold'),
 ('davis_cold','Davis','cold'),('kiba_official','KIBA','random'),
 ('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','cold'),
 ('kinase_temporal','Kinase','temporal'),('kinase_random','Kinase','random'),
 ('kinase_scaffold','Kinase','scaffold'),('kinase_target','Kinase','cold')]

rows = []
for cell, ds, pr in CELLS:
    p2 = 'results/p2/%s' % cell
    if not os.path.isdir(p2): continue
    ms = avail('results/p2', cell)
    if not ms: continue
    V = {}; T = {}
    for m in ms:
        v = pd.read_parquet('%s/pred_%s_val.parquet' % (p2, m))
        t = pd.read_parquet('%s/pred_%s_test.parquet' % (p2, m))
        V[m] = (v['pred'].values.astype(float), v['pK'].values.astype(float))
        T[m] = (t['pred'].values.astype(float), t['pK'].values.astype(float))
    yv = V[ms[0]][1]; yt = T[ms[0]][1]
    Xv = np.column_stack([V[m][0] for m in ms]); Xt = np.column_stack([T[m][0] for m in ms])
    mv = np.isfinite(Xv).all(1) & np.isfinite(yv); Xv, yv = Xv[mv], yv[mv]
    mt = np.isfinite(Xt).all(1) & np.isfinite(yt); Xt, yt = Xt[mt], yt[mt]
    if len(yt) < 50: continue                                     # 测试集太小，跳过 cv5
    p_ours = cv5_pred(Xt, yt)
    our = dict(rmse=rmse(p_ours, yt), mse=float(np.mean((p_ours-yt)**2)),
               ci=float(ci_harrell(p_ours, yt)))
    tr = {m: rmse(T[m][0], T[m][1]) for m in ms}
    opp = min(tr, key=tr.get)
    tb = np.isfinite(T[opp][0]) & np.isfinite(T[opp][1])
    po = T[opp][0][tb]; yo = T[opp][1][tb]
    oth = dict(rmse=rmse(po, yo), mse=float(np.mean((po-yo)**2)),
               ci=float(ci_harrell(po, yo)))
    rows.append(dict(cell=cell, ds=ds, proto=pr, K=len(ms), n=int(len(yt)),
                     ours=our, opp=oth, opp_name=opp,
                     win_rmse=bool(our['rmse'] < oth['rmse']),
                     win_mse=bool(our['mse'] < oth['mse']),
                     win_ci=bool(our['ci'] > oth['ci'])))

n = len(rows)
print('%-18s %4s %7s | %8s %8s %8s | %8s %8s %8s | %s' % (
    'cell','K','n','oRMSE','oMSE','oCI','xRMSE','xMSE','xCI','W'))
for r in rows:
    o, x = r['ours'], r['opp']
    w = ''.join(k[4].upper() if v else '-' for k, v in
                [('win_rmse', r['win_rmse']), ('win_mse', r['win_mse']), ('win_ci', r['win_ci'])])
    print('%-18s %4d %7d | %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f | %s' % (
        r['cell'], r['K'], r['n'], o['rmse'], o['mse'], o['ci'],
        x['rmse'], x['mse'], x['ci'], w))
print()
print('=== cv5 口径，%d 格，vs 池内最佳单成员 ===' % n)
for k, lab in [('win_rmse','RMSE'), ('win_mse','MSE'), ('win_ci','CI')]:
    print('  %-5s 第一: %d/%d' % (lab, sum(r[k] for r in rows), n))
print('  三项全第一: %d/%d' % (sum(r['win_rmse'] and r['win_mse'] and r['win_ci'] for r in rows), n))
json.dump(rows, open('results/strengthen/cv5_three_metrics.json','w'), indent=1)
print('\nWROTE results/strengthen/cv5_three_metrics.json')