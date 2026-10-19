# -*- coding: utf-8 -*-
"""统一原创层验证：LOD(检测下限)感知聚合 + 反稀释收缩
在全部 14 格上检验：(a) 是否安全 (b) 能否把三红格变正
对手双口径：strict(test-oracle) vs fair(val-selected)"""
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
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]

def fit_lod(X,y,floor,K):
    cens=np.abs(y-floor)<1e-6
    def obj(th):
        w=np.abs(th[:K]); w=w/(w.sum()+1e-12)
        e=X@w+th[K]-y
        return float(np.mean(np.where(cens,np.maximum(e,0.0),e)**2))
    r_=minimize(obj,np.r_[np.ones(K)/K,float(y.mean())],method='Nelder-Mead',
                options={'maxiter':60000,'xatol':1e-7,'fatol':1e-11})
    w=np.abs(r_.x[:K]); return w/(w.sum()+1e-12), r_.x[K]
def detect_floor(y):
    vv,cc=np.unique(np.round(y,6),return_counts=True); i=int(cc.argmax())
    return (float(vv[i]), float(cc[i]/len(y))) if cc[i]/len(y)>0.05 else (None,0.0)

rows=[];ok=0
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
    vr=np.array([rmse(Xv[:,i],yv) for i in range(K)]); tr=np.array([rmse(Xt[:,i],yt) for i in range(K)])
    vb=int(vr.argmin())
    sim=rmse(apply_th(fit_simplex(Xv,yv),Xt),yt)
    uni=rmse(Xt.mean(1),yt); med=rmse(np.median(Xt,1),yt)
    floor,cf=detect_floor(yv)
    lod=np.nan
    if floor is not None:
        w,c=fit_lod(Xv,yv,floor,K); lod=rmse(Xt@w+c,yt)
    r={'cell':cell,'K':K,'floor':floor,'cens_frac':cf,
       'strict_opp':float(tr.min()),'fair_opp':float(tr[vb]),'fair_name':ms[vb],
       'matrix_ours':row['ours']['rmse'],'deployed_simplex':sim,'uniform':uni,'median':med,'lod':lod}
    r['strict_win']={'simplex':bool(sim<tr.min()),'uniform':bool(uni<tr.min()),'median':bool(med<tr.min()),
                     'lod':bool(lod<tr.min()) if lod==lod else False}
    r['fair_win']={'simplex':bool(sim<tr[vb]),'uniform':bool(uni<tr[vb]),'median':bool(med<tr[vb]),
                   'lod':bool(lod<tr[vb]) if lod==lod else False}
    rows.append(r); ok+=1
    print('%-18s K=%2d floor=%-6s cens=%4.0f%% | strict %.4f fair %.4f | sim %.4f uni %.4f med %.4f lod %.4f | fair赢:%s'%(
        cell,K,floor,cf*100,tr.min(),tr[vb],sim,uni,med,lod,
        ''.join([k[0] for k,v in r['fair_win'].items() if v])))
json.dump(rows,open('results/strengthen/unified_layer.json','w'),indent=1)
print()
print('=== 汇总（%d 格）==='%ok)
for k in ['simplex','uniform','median','lod']:
    s=sum(r['strict_win'][k] for r in rows if not (r[k]!=r[k]))
    f=sum(r['fair_win'][k] for r in rows if not (r[k]!=r[k]))
    tot=sum(1 for r in rows if not (r[k]!=r[k]))
    print('  %-8s 严格赢 %2d/%d   公平赢 %2d/%d'%(k,s,tot,f,tot))
print()
print('=== LOD 层 vs 已部署（增量，负=更好）===')
for r in rows:
    if r['lod']==r['lod']:
        print('  %-18s lod-simplex = %+.4f'%(r['cell'],r['lod']-r['deployed_simplex']))
