# -*- coding: utf-8 -*-
"""三格完整裁决：严格口径(test-oracle对手) vs 公平口径(val选对手) × 各方案
并全格检验 LOD 感知层是否全局安全"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
from scipy.optimize import minimize
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
def lod_fit(X,y,floor,K):
    cens=np.abs(y-floor)<1e-6
    def obj(th):
        w=np.abs(th[:K]); w=w/(w.sum()+1e-12)
        e=Xv_=X@w+th[K]-y
        e2=np.where(cens,np.maximum(e,0.0),e)
        return float(np.mean(e2**2))
    r_=minimize(obj,np.r_[np.ones(K)/K,float(y.mean())],method='Nelder-Mead',
                options={'maxiter':40000,'xatol':1e-6,'fatol':1e-10})
    w=np.abs(r_.x[:K]); return w/(w.sum()+1e-12), r_.x[K]
CFG={'davis_scaffold':('Davis','scaffold'),'kiba_scaffold':('KIBA','scaffold'),'kiba_cold':('KIBA','冷启动')}
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
final={}
print('='*100)
for cell,(ds,pr) in CFG.items():
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
    row=M[(ds,pr)]
    vr=np.array([rmse(Xv[:,i],yv) for i in range(K)])
    tr=np.array([rmse(Xt[:,i],yt) for i in range(K)])
    vb, tb = int(vr.argmin()), int(tr.argmin())
    th_sim=fit_simplex(Xv,yv); sim=rmse(apply_th(th_sim,Xt),yt)
    th_nnls=fit_nnls(Xv,yv); nn=rmse(apply_th(th_nnls,Xt),yt)
    uni=rmse(Xt.mean(1),yt); med=rmse(np.median(Xt,1),yt)
    # LOD
    vv,cc=np.unique(np.round(yv,6),return_counts=True); fi=int(cc.argmax())
    floor=float(vv[fi]) if cc[fi]/len(yv)>0.05 else None
    lod=np.nan
    if floor is not None:
        w,c=lod_fit(Xv,yv,floor,K); lod=rmse(Xt@w+c,yt)
    R={'K':K,
       'test_oracle_opp':float(tr.min()),'test_oracle_name':ms[tb],
       'val_selected_opp':float(tr[vb]),'val_selected_name':ms[vb],'val_sel_valRMSE':float(vr[vb]),
       'matrix_ours':row['ours']['rmse'],
       'deployed_simplex':sim,'deployed_nnls':nn,'uniform':uni,'median':med,
       'lod_floor':floor,'lod_aware':lod}
    R['strict_vs_oracle']={'simplex':bool(sim<tr.min()),'lod':bool(lod< tr.min()) if lod==lod else False,
                           'uniform':bool(uni<tr.min()),'median':bool(med<tr.min())}
    R['fair_vs_valsel']={'simplex':bool(sim<tr[vb]),'lod':bool(lod<tr[vb]) if lod==lod else False,
                         'uniform':bool(uni<tr[vb]),'median':bool(med<tr[vb])}
    final[cell]=R
    print('### %s  K=%d'%(cell,K))
    print('  严格对手(test-oracle)= %s %.4f'%(ms[tb],tr.min()))
    print('  公平对手(val-selected)= %s %.4f  (val选它时 valRMSE=%.4f)'%(ms[vb],tr[vb],vr[vb]))
    print('  候选: simplex %.4f | NNLS %.4f | uniform %.4f | median %.4f | LOD %.4f'%(sim,nn,uni,med,lod))
    print('  => 严格口径可赢者: %s'%[k for k,v in R['strict_vs_oracle'].items() if v])
    print('  => 公平口径可赢者: %s'%[k for k,v in R['fair_vs_valsel'].items() if v])
    print()
json.dump(final,open('results/strengthen/three_cells_verdict.json','w'),indent=1)
print('WROTE three_cells_verdict.json')
