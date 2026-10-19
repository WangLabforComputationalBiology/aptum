# -*- coding: utf-8 -*-
"""CATS-GATED：删失感知层 + 无标签门控（真原创层，全局安全性验证）
门控：val 检测到的地板占比 > tau 才启用 binned-Tobit 变体；否则标准 simplex。
在 14 格上验证：相对部署口径是否全局不伤。"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
from scipy.optimize import nnls
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex, apply_th
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def memb(cell):
    r=[]
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)):
        m=os.path.basename(f).replace('pred_','').replace('_test.parquet','')
        if 'kairos' in m: continue
        if os.path.exists('results/p2/%s/pred_%s_val.parquet'%(cell,m)): r.append(m)
    return r
def detect(y,minf=0.05):
    vv,cc=np.unique(np.round(y,6),return_counts=True); i=int(cc.argmax())
    return (float(vv[i]),float(cc[i]/len(y))) if cc[i]/len(y)>minf else (None,0.0)
def simplex_proj(w):
    w=np.clip(w,0,None); s=w.sum(); return w/s if s>1e-12 else np.ones(len(w))/len(w)
def cats_fit(X,y,floor,K,iters=40):
    """删失感知堆叠：地板行用 hinge（只罚过预测），配合 simplex 投影 + 有界截距"""
    cens=np.abs(y-floor)<1e-6
    th=simplex_proj(np.ones(K)/K); c=float(y[~cens].mean()) if (~cens).sum() else float(y.mean())
    for _ in range(iters):
        yy=y.copy()
        mu=X@th+c
        if cens.sum():
            # IRLS 权重：删失行只在下侧惩罚 → 过预测时权重大
            e=mu[cens]-floor
            wgt=np.where(e>0, np.clip(1.0+2.0*e,1.0,4.0), 0.25)
            yy[cens]=floor + np.where(e>0, e*0.0, np.clip(e,-1.0,0.0))
            sw=np.ones(len(y)); sw[cens]=wgt
        else:
            sw=np.ones(len(y))
        A=np.hstack([X,np.ones((len(X),1))])*np.sqrt(sw)[:,None]
        b=yy*np.sqrt(sw)
        sol,_=nnls(A.astype(np.float64),np.asarray(b,float))
        th=simplex_proj(sol[:K]); c=float(sol[K])
    return th,c
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
TAUS=[0.20,0.30,0.40,0.50]
rows=[]
print('%-18s %4s %5s %5s | %8s %8s %8s | %s'%('cell','K','floor','cens','部署','CATS','门控后','启用?'))
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
    dep=rmse(apply_th(fit_simplex(Xv,yv),Xt),yt)
    floor,cf=detect(yv)
    cats=np.nan; gated=dep; used=False
    if floor is not None:
        th,c=cats_fit(Xv,yv,floor,K); cats=rmse(Xt@th+c,yt)
        if cf>0.30: gated=cats; used=True
    rows.append({'cell':cell,'K':K,'floor':floor,'cens_frac':cf,'deployed':dep,'cats':cats,
                 'gated':gated,'gate_used':used,'delta':gated-dep if gated==gated else 0.0})
    print('%-18s %4d %5s %4.0f%% | %8.4f %8.4f %8.4f | %s'%(cell,K,floor,cf*100,dep,
        cats if cats==cats else float('nan'),gated,'启用' if used else '—'))
json.dump(rows,open('results/strengthen/cats_gated.json','w'),indent=1)
d=[r['delta'] for r in rows]
print()
print('门控(tau=0.30)全局: 中位 %+.4f  改好 %d/%d  启用 %d 格'%(
    np.median(d),sum(1 for x in d if x<0),len(d),sum(r['gate_used'] for r in rows)))
