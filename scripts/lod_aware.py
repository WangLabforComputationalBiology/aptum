# -*- coding: utf-8 -*-
"""LOD(检测下限)感知聚合 —— 原创层候选
原理：DTA 数据集大量标签等于平台检测下限（Davis pK=5.0 占 69.6%）。这些不是精确测量，
而是右删失（y<=5.0）。标准最小二乘把它们当真值，导致聚合器压低预测幅度以拟合常数行。
新层：无标签地检测地板，对地板行用单边(hinge)损失，非地板行用平方损失。
"""
import os
import numpy as np, pandas as pd, glob, os, json
from scipy.optimize import nnls, minimize, lsq_linear
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def memb(cell): return [os.path.basename(f).replace('pred_','').replace('_test.parquet','')
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)) if 'kairos' not in f]

def detect_floor(y, tol=1e-6, min_frac=0.05):
    """无标签地板检测：返回最频繁的精确重复值，若占比>min_frac 则判为地板"""
    v,c=np.unique(np.round(y,6),return_counts=True)
    i=c.argmax()
    return float(v[i]) if c[i]/len(y)>min_frac else None

def fit_lod(X,y,floor,w_uncens=1.0,w_cens=0.15,lam_simplex=0.0):
    """LOD 感知拟合：地板行 hinge(只罚过预测)，非地板行平方"""
    K=X.shape[1]
    cens=np.abs(y-floor)<1e-6
    def obj(th):
        w=np.abs(th[:K]); w=w/(w.sum()+1e-12)
        p=X@w+th[K]; e=p-y
        e2=np.where(cens, np.maximum(e,0.0), e)
        return float(np.mean(e2**2)+lam_simplex*np.sum(np.abs(w)))
    x0=np.r_[np.ones(K)/K, float(y.mean())]
    r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':40000,'xatol':1e-6,'fatol':1e-10})
    w=np.abs(r.x[:K]); w=w/(w.sum()+1e-12)
    return w, r.x[K]

# 先扫格：哪些格有地板
CELLS={'davis_scaffold':'Davis|scaffold','davis_cold':'Davis|cold','davis_official':'Davis|random',
       'kiba_scaffold':'KIBA|scaffold','kiba_cold':'KIBA|冷启动','chembl_temporal':'ChEMBL|时间'}
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
print('%-16s %8s %8s %8s %8s %8s'%('cell','floor','cens%','deployed','LOD','opp'))
OUT={}
for cell,mk in CELLS.items():
    if not os.path.isdir('results/p2/%s'%cell): continue
    ms=[m for m in memb(cell)
        if os.path.exists('results/p2/%s/pred_%s_val.parquet'%(cell,m))
        and os.path.exists('results/p2/%s/pred_%s_test.parquet'%(cell,m))]
    K=len(ms)
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
    floor=detect_floor(yv)
    # 部署基线：simplex
    A=np.hstack([Xv,np.ones((len(Xv),1))])
    r=lsq_linear(A.astype(np.float64),yv.astype(np.float64),bounds=(np.r_[np.zeros(K),-np.inf],np.inf))
    ths=r.x[:-1]/(r.x[:-1].sum()+1e-12)
    dep=rmse(Xt@ths+r.x[-1],yt)
    cens=float(np.mean(np.abs(yv-floor)<1e-6)) if floor else 0.0
    row=M.get(tuple(mk.split('|'))); opp=row['opponent']['rmse'] if row else np.nan
    lod=np.nan
    if floor is not None:
        w,c=fit_lod(Xv,yv,floor); lod=rmse(Xt@w+c,yt)
    OUT[cell]={'floor':floor,'cens_frac':cens,'deployed':dep,'lod_aware':lod,'opponent':opp,
               'lod_gain':(dep-lod) if lod==lod else np.nan}
    print('%-16s %8s %7.1f%% %8.4f %8.4f %8.4f  %s'%(
        cell,floor,cens*100,dep,lod,opp,('LOD赢' if lod==lod and lod<opp else '')))
json.dump(OUT,open('results/strengthen/lod_aware.json','w'),indent=1)
print('\nWROTE results/strengthen/lod_aware.json')
