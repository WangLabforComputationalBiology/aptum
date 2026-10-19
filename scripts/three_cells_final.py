# -*- coding: utf-8 -*-
"""三格正向修复（冻结口径）：用 kairos_rule.py 的 fit_nnls/fit_simplex
对照：无标签可达基线（val 选最优单成员、uniform、median）vs 论文的 test-oracle 对手
新增原创层候选：LOD 感知（检测下限 hinge）"""
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
CFG={'davis_scaffold':('Davis','scaffold'),'kiba_scaffold':('KIBA','scaffold'),'kiba_cold':('KIBA','冷启动')}
OUT={}
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
    row=M[(ds,pr)]; opp=row['opponent']['rmse']; oppname=row['opponent']['name']
    oursmat=row['ours']['rmse']
    # 冻结口径基线
    th_nnls=fit_nnls(Xv,yv); dep_nnls=rmse(apply_th(th_nnls,Xt),yt)
    th_sim=fit_simplex(Xv,yv); dep_sim=rmse(apply_th(th_sim,Xt),yt)
    vr=np.array([rmse(Xv[:,i],yv) for i in range(K)])
    vb=int(vr.argmin())
    R={'K':K,'n_val':len(yv),'n_test':len(yt),'opponent_name':oppname,'opponent_rmse':opp,
       'matrix_ours':oursmat,'deployed_nnls':dep_nnls,'deployed_simplex':dep_sim,
       'uniform':rmse(Xt.mean(1),yt),'median':rmse(np.median(Xt,1),yt),
       'val_best_name':ms[vb],'val_best_rmse':rmse(Xt[:,vb],yt),
       'test_oracle_rmse':float(vr.min()*0+min(rmse(Xt[:,i],yt) for i in range(K)))}
    # LOD 感知
    yv_r=np.round(yv,6); vv,cc=np.unique(yv_r,return_counts=True)
    fi=cc.argmax(); floor=float(vv[fi]) if cc[fi]/len(yv_r)>0.05 else None
    R['lod_floor']=floor; R['lod_cens_frac']=float(cc[fi]/len(yv_r)) if floor else 0.0
    if floor is not None:
        cens=np.abs(yv-floor)<1e-6
        def obj(th):
            w=np.abs(th[:K]); w=w/(w.sum()+1e-12)
            e=Xv@w+th[K]-yv
            e2=np.where(cens,np.maximum(e,0.0),e)
            return float(np.mean(e2**2))
        r_=minimize(obj,np.r_[np.ones(K)/K,float(yv.mean())],method='Nelder-Mead',
                    options={'maxiter':40000,'xatol':1e-6,'fatol':1e-10})
        w=np.abs(r_.x[:K]); w=w/(w.sum()+1e-12)
        R['lod_aware']=rmse(Xt@w+r_.x[K],yt)
    # 判读
    R['vs_opponent_nnls']='赢' if dep_nnls<opp else '输'
    R['val_best_vs_opponent']='赢' if R['val_best_rmse']<opp else '输'
    OUT[cell]=R
    print('=== %s  K=%d ==='%(cell,K))
    print('  论文对手=%-12s %.4f  (test-oracle=%.4f)'%(oppname,opp,R['test_oracle_rmse']))
    print('  矩阵值 %.4f'%oursmat)
    print('  无标签可达: val-best单成员(%s) %.4f | uniform %.4f | median %.4f'%(
        ms[vb],R['val_best_rmse'],R['uniform'],R['median']))
    print('  部署: NNLS %.4f | simplex %.4f'%(dep_nnls,dep_sim))
    if floor: print('  LOD感知(floor=%.1f, 删失%.0f%%): %.4f'%(floor,R['lod_cens_frac']*100,R['lod_aware']))
json.dump(OUT,open('results/strengthen/three_cells_final.json','w'),indent=1)
print('\nWROTE three_cells_final.json')
