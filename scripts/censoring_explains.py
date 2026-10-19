# -*- coding: utf-8 -*-
"""决定性检验：删失是否【解释】了退化解谜题（这才是原创发现的真价值）"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex, fit_nnls
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
cell='davis_scaffold'
ms=[os.path.basename(f).replace('pred_','').replace('_test.parquet','')
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)) if 'kairos' not in f]
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
print('=== 全部 val 行（含删失）======')
th=fit_simplex(Xv,yv); p=Xt@th[1:]+th[0]
print('  θ 非零:',{ms[i]:round(float(th[1+i]),3) for i in range(len(ms)) if th[1+i]>1e-6},' 截距 %.3f'%th[0])
print('  预测 sd/标签 sd = %.3f   test RMSE %.4f'%(p.std()/yt.std(),rmse(p,yt)))
unc=np.abs(yv-5.0)>1e-9
print()
print('=== 只在【非删失】val 行上拟合（%d/%d 行）======'%(unc.sum(),len(yv)))
th2=fit_simplex(Xv[unc],yv[unc]); p2=Xt@th2[1:]+th2[0]
print('  θ 非零:',{ms[i]:round(float(th2[1+i]),3) for i in range(len(ms)) if th2[1+i]>1e-6},' 截距 %.3f'%th2[0])
print('  预测 sd/标签 sd = %.3f   test RMSE %.4f'%(p2.std()/yt.std(),rmse(p2,yt)))
print()
print('=== 只在非删失【test】行上评估（%d 行）======'%unc.sum())
print('  含删失拟合: %.4f | 非删失拟合: %.4f'%(rmse(p[~np.abs(yt-5.0)<1e-9],yt[~np.abs(yt-5.0)<1e-9]),
      rmse(p2[~np.abs(yt-5.0)<1e-9],yt[~np.abs(yt-5.0)<1e-9])))
print()
print('=== 判读 ===')
print('  含删失拟合的振幅比: %.3f'%(p.std()/yt.std()))
print('  非删失拟合的振幅比: %.3f'%(p2.std()/yt.std()))
print('  → 振幅收缩是否由删失造成: %s'%('是' if p2.std()>p.std()*1.3 else '不完全'))
