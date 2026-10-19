# -*- coding: utf-8 -*-
"""检测下限感知聚合 (Floor-Aware Aggregation, FAA) —— 纯原创层
事实：DTA 标签大面积 = 平台检测下限（Davis 69.6% 恰为 5.0），是右删失而非测量值。
标准 MSE 把地板行当真值 → 权重被拉向复现常数（这正是论文 §4.3(b) 退化解的真因）。
本层：地板行只用单边损失（只罚过预测），非地板行用平方损失。无自由超参、无标签。
"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def memb(cell):
    r=[]
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)):
        m=os.path.basename(f).replace('pred_','').replace('_test.parquet','')
        if 'kairos' in m: continue
        if os.path.exists('results/p2/%s/pred_%s_val.parquet'%(cell,m)): r.append(m)
    return r

def detect_floor(y, min_frac=0.05):
    v,c = np.unique(np.round(y,6), return_counts=True)
    i = int(c.argmax())
    return (float(v[i]), float(c[i]/len(y))) if c[i]/len(y) >= min_frac else (None, 0.0)

def fit_faa(X, y, floor, K, mode='hinge'):
    """地板感知拟合：simplex 约束 θ + 自由截距"""
    cens = np.abs(y-floor) < 1e-6
    def unpack(p):
        th = np.clip(p[:K], 0, None); s = th.sum()
        return (th/s if s > 1e-12 else np.ones(K)/K), p[K]
    def obj(p):
        th, c = unpack(p); mu = X@th + c; e = mu - y
        if mode == 'hinge':
            return float(np.mean(np.where(cens, np.maximum(e, 0.0), e)**2))
        # tobit: 删失行按 P(Y<=floor) 计入似然
        sig = max(np.sqrt(np.mean(e[~cens]**2)) if (~cens).sum() else 1.0, 1e-3)
        nll = 0.0
        if (~cens).sum(): nll += 0.5*np.sum((e[~cens]/sig)**2)
        if cens.sum(): nll += -np.sum(norm.logcdf((floor - mu[cens])/sig))
        return float(nll)
    x0 = np.r_[np.ones(K)/K, float(y.mean())]
    r = minimize(obj, x0, method='Powell', options={'maxiter':20000,'xtol':1e-5,'ftol':1e-9})
    th, c = unpack(r.x)
    return th, c

CELLS=[('chembl_temporal','ChEMBL','时间'),('chembl_random','ChEMBL','随机'),
 ('chembl_scaffold','ChEMBL','scaffold'),('chembl_cold','ChEMBL','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
rows=[]
print('%-18s %3s %6s %6s | %9s %9s %9s'%('cell','K','floor','cens%','simplex','FAA-hinge','FAA-tobit'))
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
    th=fit_simplex(Xv,yv); base=rmse(Xt@th[1:]+th[0],yt)
    floor,cf=detect_floor(yv)
    h=t=np.nan
    if floor is not None:
        thh,ch=fit_faa(Xv,yv,floor,K,'hinge'); h=rmse(Xt@thh+ch,yt)
        tht,ct=fit_faa(Xv,yv,floor,K,'tobit'); t=rmse(Xt@tht+ct,yt)
    rows.append({'cell':cell,'K':K,'floor':floor,'cens_frac':cf,'simplex':base,
                 'faa_hinge':h,'faa_tobit':t})
    print('%-18s %3d %6s %5.0f%% | %9.4f %9s %9s'%(cell,K,floor,cf*100,base,
        ('%.4f'%h) if h==h else '  n/a', ('%.4f'%t) if t==t else '  n/a'))
json.dump(rows,open('results/strengthen/floorlayer.json','w'),indent=1)
print()
f=[r for r in rows if r['faa_hinge']==r['faa_hinge']]
if f:
    d=[r['faa_hinge']-r['simplex'] for r in f]
    print('FAA-hinge vs simplex（有地板的格）: 中位 %+.4f  改好 %d/%d'%(np.median(d),sum(1 for x in d if x<0),len(d)))
    for r in f: print('   %-18s %+.4f  %s'%(r['cell'],r['faa_hinge']-r['simplex'],'✓' if r['faa_hinge']<r['simplex'] else '✗'))
