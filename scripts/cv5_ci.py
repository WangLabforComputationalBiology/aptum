# -*- coding: utf-8 -*-
"""cv5 口径的全格 RMSE + Harrell CI（主表用），用冻结的 ci_lib"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex
from ci_lib import ci_harrell
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def memb(cell):
    r=[]
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)):
        m=os.path.basename(f).replace('pred_','').replace('_test.parquet','')
        if 'kairos' in m: continue
        if os.path.exists('results/p2/%s/pred_%s_val.parquet'%(cell,m)): r.append(m)
    return r
def cv5(X,y,seed=0,nf=5):
    n=len(y); rng=np.random.default_rng(seed); idx=rng.permutation(n); p=np.zeros(n)
    for f in np.array_split(idx,nf):
        t=np.setdiff1d(np.arange(n),f); th=fit_simplex(X[t],y[t]); p[f]=X[f]@th[1:]+th[0]
    return p
CELLS=[('chembl_temporal','ChEMBL','时间'),('chembl_random','ChEMBL','随机'),
 ('chembl_scaffold','ChEMBL','scaffold'),('chembl_cold','ChEMBL','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
out={}
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
    p=cv5(Xt,yt)
    out[cell]={'K':K,'rmse':rmse(p,yt),'ci':float(ci_harrell(p,yt)),'n':int(len(yt))}
    print('%-18s K=%2d rmse=%.4f ci=%.4f n=%d'%(cell,K,out[cell]['rmse'],out[cell]['ci'],out[cell]['n']))
json.dump(out,open('results/strengthen/cv5_ci.json','w'),indent=1)
print('\nWROTE cv5_ci.json')
