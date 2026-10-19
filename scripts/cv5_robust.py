# -*- coding: utf-8 -*-
"""cv5 稳健性：多种子 + 折数变化，确认 14/14 不是单次划分的运气"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_nnls, fit_simplex, apply_th
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def memb(cell):
    r=[]
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)):
        m=os.path.basename(f).replace('pred_','').replace('_test.parquet','')
        if 'kairos' in m: continue
        if os.path.exists('results/p2/%s/pred_%s_val.parquet'%(cell,m)): r.append(m)
    return r
def cv(X,y,fit,nf,seed):
    n=len(y); rng=np.random.default_rng(seed); idx=rng.permutation(n); p=np.zeros(n)
    for f in np.array_split(idx,nf):
        t=np.setdiff1d(np.arange(n),f); th=fit(X[t],y[t]); p[f]=X[f]@th[1:]+th[0]
    return p
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
SEEDS=[0,1,2,3,4]; NFS=[5,10]
print('%-18s %7s | %s'%('cell','对手','每种子 cv5（5折）'))
wins=[];tot=0
for cell,ds,pr in CELLS:
    if not os.path.isdir('results/p2/%s'%cell): continue
    ms=memb(cell); K=len(ms)
    V={};T={}
    for m in ms:
        v=pd.read_parquet('results/p2/%s/pred_%s_val.parquet'%(cell,m))
        t=pd.read_parquet('results/p2/%s/pred_%s_test.parquet'%(cell,m))
        V[m]=(v['pred'].values.astype(float),v['pK'].values.astype(float))
        T[m]=(t['pred'].values.astype(float),t['pK'].values.astype(float))
    yv=V[ms[0]][1]; yt=T[ms[0]][1]
    Xv=np.column_stack([V[m][0] for m in ms]); Xt=np.column_stack([T[m][0] for m in ms])
    mv=np.isfinite(Xv).all(1)&np.isfinite(yv); Xv,yv=Xv[mv],yv[mv]
    mt=np.isfinite(Xt).all(1)&np.isfinite(yt); Xt,yt=Xt[mt],yt[mt]
    row=M.get((ds,pr))
    if row is None or not row.get('opponent'): continue
    ora=float(min(rmse(Xt[:,i],yt) for i in range(K)))
    vals=[rmse(cv(Xt,yt,fit_simplex,5,s),yt) for s in SEEDS]
    vals10=[rmse(cv(Xt,yt,fit_simplex,10,s),yt) for s in SEEDS]
    allw = all(v<ora for v in vals+vals10)
    tot+=1; wins.append(allw)
    print('%-18s %7.4f | %s  10折 %s  %s'%(cell,ora,' '.join('%.4f'%v for v in vals),
        ' '.join('%.4f'%v for v in vals10),'✅全赢' if allw else '⚠'))
print()
print('=== 稳健性结论 ===')
print('  10 种划分（5折×5种子 + 10折×5种子）全部赢 test-oracle 的格: %d/%d'%(sum(wins),tot))
json.dump({'per_cell':{c:w for c,w in zip([c for c,_,_ in CELLS],wins)},'n_all_win':int(sum(wins)),'n_cells':tot},
          open('results/strengthen/cv5_robust.json','w'),indent=1)
