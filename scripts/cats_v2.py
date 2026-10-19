# -*- coding: utf-8 -*-
"""CATS v2：删失感知 Tobit 堆叠（EM 版本）+ 无标签门控
EM 正确性：E 步对删失行用截断高斯期望填补，M 步加权最小二乘 + σ 更新。
门控（无标签）：val 上检测到的地板占比 cens_frac > 阈值 才启用；否则退化为标准 simplex。
在 14 格上验证：对 deployed 与 cv5 两个基线分别叠加。"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
from scipy.optimize import nnls
from scipy.stats import norm
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
def detect(y,tol=1e-6,minf=0.05):
    vv,cc=np.unique(np.round(y,6),return_counts=True); i=int(cc.argmax())
    return (float(vv[i]),float(cc[i]/len(y))) if cc[i]/len(y)>minf else (None,0.0)
def simplex_proj(w):
    w=np.clip(w,0,None); s=w.sum(); return w/s if s>1e-12 else np.ones(len(w))/len(w)
def em_tobit(X,y,floor,K,iters=60):
    """EM for censored Gaussian; simplex-constrained θ + free intercept + σ"""
    cens=np.abs(y-floor)<1e-6
    th=simplex_proj(np.ones(K)/K); c=float(y[~cens].mean()) if (~cens).sum() else float(y.mean())
    s=float(y.std())
    yy=y.copy()
    for _ in range(iters):
        mu=X@th+c
        if cens.sum():
            a=(floor-mu[cens])/s
            yy[cens]=mu[cens]-s*norm.pdf(a)/np.maximum(norm.cdf(a),1e-12)  # E-step
        A=np.hstack([X,np.ones((len(X),1))])
        sol,_=nnls(A.astype(np.float64),np.asarray(yy,float))
        th=simplex_proj(sol[:K]); c=float(sol[K])
        res=yy-X@th-c
        s=float(np.sqrt(max(np.mean(res**2),1e-6)))
    return th,c
def cv_fit(X,y,fit):
    n=len(y); rng=np.random.default_rng(0); idx=rng.permutation(n)
    p=np.zeros(n)
    for f in np.array_split(idx,5):
        t=np.setdiff1d(np.arange(n),f); r=fit(X[t],y[t]); p[f]=X[f]@r[0]+r[1]
    return p
def sim_fit(X,y): 
    th=fit_simplex(X,y); return th[1:],th[0]
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
rows=[]
print('%-18s %3s %5s %5s | %7s %7s %7s %7s | %7s %7s'%('cell','K','floor','cens','对手','dep','depCATS','cv5','cv5CATS','cv5赢'))
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
    dep=rmse(apply_th(fit_simplex(Xv,yv),Xt),yt)
    c5=rmse(cv_fit(Xt,yt,sim_fit),yt)
    floor,cf=detect(yv)
    depC=c5C=np.nan
    if floor is not None and cf>0.0:
        th,c=em_tobit(Xv,yv,floor,K); depC=rmse(Xt@th+c,yt)
        c5C=rmse(cv_fit(Xt,yt,lambda A,b: em_tobit(A,b,floor,A.shape[1])),yt)
    r={'cell':cell,'K':K,'floor':floor,'cens_frac':cf,'opponent_test_oracle':ora,
       'deployed':dep,'deployed_cats':depC,'cv5':c5,'cv5_cats':c5C,
       'cv5_win':bool(c5<ora),'cv5cats_win':bool(c5C<ora) if c5C==c5C else False}
    rows.append(r)
    print('%-18s %3d %5s %4.0f%% | %7.4f %7.4f %7s %7.4f %7s | %7s'%(
        cell,K,floor,cf*100,ora,dep,('%.4f'%depC) if depC==depC else '   n/a',
        c5,('%.4f'%c5C) if c5C==c5C else '   n/a','✅' if r['cv5_win'] else '  '))
json.dump(rows,open('results/strengthen/cats_v2.json','w'),indent=1)
print()
print('=== 汇总 ===')
print('  cv5 对 test-oracle 赢 %d/%d'%(sum(r['cv5_win'] for r in rows),len(rows)))
v=[r for r in rows if r['cv5cats_win'] or (r['cv5_cats']==r['cv5_cats'])]
print('  cv5+CATS 对 test-oracle 赢 %d/%d'%(sum(r['cv5cats_win'] for r in rows if r['cv5_cats']==r['cv5_cats']),len(v)))
d=[r['deployed_cats']-r['deployed'] for r in rows if r['deployed_cats']==r['deployed_cats']]
print('  CATS vs deployed: 中位 %+.4f  改好 %d/%d'%(np.median(d),sum(1 for x in d if x<0),len(d)))
d2=[r['cv5_cats']-r['cv5'] for r in rows if r['cv5_cats']==r['cv5_cats']]
print('  CATS vs cv5:      中位 %+.4f  改好 %d/%d'%(np.median(d2),sum(1 for x in d2 if x<0),len(d2)))
