# -*- coding: utf-8 -*-
"""原创层：删失感知 Tobit 堆叠 (Censoring-Aware Tobit Stacking, CATS)
统计事实：DTA 数据大量标签 = 平台检测下限（Davis 69.6% 恰为 5.0），是右删失。
标准最小二乘把删失行当真值 → 权重被拉向地板（正是论文 §4.3(b) 观察到的"振幅收缩到 18%"）。
正解：用删失高斯似然拟合 θ（Tobit 1958），无自由超参（σ 联合估计）。
检验：6 个有地板的格上，CATS vs 部署 simplex。"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
from scipy.optimize import minimize
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
def tobit_fit(X,y,floor,K):
    cens=np.abs(y-floor)<1e-6; unc=~cens
    def nll(p):
        z=p[:K]; s=np.exp(p[K+1])
        w=np.exp(z-z.max()); w/=w.sum()
        mu=X@w+p[K]
        n=0.0
        if unc.sum(): r=(y[unc]-mu[unc])/s; n+=0.5*np.sum(r*r)+unc.sum()*p[K+1]
        if cens.sum(): q=(floor-mu[cens])/s; n+=-np.sum(norm.logcdf(q))
        return n
    p0=np.r_[np.zeros(K),float(y.mean()),np.log(float(y.std()))]
    res=minimize(nll,p0,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12})
    w=np.exp(res.x[:K]-res.x[:K].max()); w/=w.sum()
    return w,res.x[K],res.fun
def detect(y):
    vv,cc=np.unique(np.round(y,6),return_counts=True); i=int(cc.argmax())
    return (float(vv[i]),float(cc[i]/len(y))) if cc[i]/len(y)>0.05 else (None,0.0)
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CFG={'davis_official':('Davis','随机'),'davis_scaffold':('Davis','scaffold'),'davis_cold':('Davis','冷启动'),
     'kiba_official':('KIBA','随机'),'kiba_scaffold':('KIBA','scaffold'),'kiba_cold':('KIBA','冷启动')}
print('%-16s %4s %6s %5s | %8s %8s %8s | %-8s %-8s'%('cell','K','floor','cens','部署','CATS','对手','CATS赢','部署赢'))
rows=[]
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
    row=M[(ds,pr)]; opp=row['opponent']['rmse']
    tr=np.array([rmse(Xt[:,i],yt) for i in range(K)]); ora=tr.min()
    dep=rmse(apply_th(fit_simplex(Xv,yv),Xt),yt)
    floor,cf=detect(yv)
    w,c,nll=tobit_fit(Xv,yv,floor,K); cats=rmse(Xt@w+c,yt)
    r={'cell':cell,'K':K,'floor':floor,'cens_frac':cf,'deployed':dep,'cats':cats,'oracle':float(ora),
       'cats_win':bool(cats<ora),'dep_win':bool(dep<ora),'cats_gain':cats-dep}
    rows.append(r)
    print('%-16s %4d %6.1f %4.0f%% | %8.4f %8.4f %8.4f | %-8s %-8s'%(cell,K,floor,cf*100,dep,cats,ora,
        '✅' if r['cats_win'] else '  ','✅' if r['dep_win'] else '  '))
json.dump(rows,open('results/strengthen/tobit_layer.json','w'),indent=1)
print()
g=[r['cats_gain'] for r in rows]
print('CATS vs 部署: 中位 %+.4f  改好 %d/%d'%(np.median(g),sum(1 for x in g if x<0),len(g)))
print('CATS 对test-oracle赢 %d/%d ; 部署对test-oracle赢 %d/%d'%(
    sum(r['cats_win'] for r in rows),len(rows),sum(r['dep_win'] for r in rows),len(rows)))
