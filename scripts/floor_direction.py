# -*- coding: utf-8 -*-
"""查清：5.0 是哪种删失方向？地板行在 RMSE 里起什么作用？"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
d=pd.read_parquet('data/splits/davis_prepared.parquet')
t=d[d['split_scaffold']=='test']
print('Davis scaffold test: n=%d  最小值 %.4f  第二小 %.4f'%(len(t),t['pK'].min(),sorted(t['pK'].unique())[1]))
print('  pK==5.0 的行数 %d (%.1f%%)'%((np.abs(t['pK']-5.0)<1e-9).sum(),100*(np.abs(t['pK']-5.0)<1e-9).mean()))
print('  分位数:',{q:round(float(t['pK'].quantile(q)),3) for q in [0,.1,.25,.5,.75,.9,1]})
print()
# 关键：论文该格的θ=0.434/0.4594 等的预测分布
cell='davis_scaffold'
ms=[os.path.basename(f).replace('pred_','').replace('_test.parquet','')
    for f in sorted(glob.glob('results/p2/%s/pred_*_test.parquet'%cell)) if 'kairos' not in f]
V={};T={}
for m in ms:
    v=pd.read_parquet('results/p2/%s/pred_%s_val.parquet'%(cell,m))
    tt=pd.read_parquet('results/p2/%s/pred_%s_test.parquet'%(cell,m))
    V[m]=(v['pred'].values.astype(float),v['pK'].values.astype(float))
    T[m]=(tt['pred'].values.astype(float),tt['pK'].values.astype(float))
yv=V[ms[0]][1]; yt=T[ms[0]][1]
Xv=np.column_stack([V[m][0] for m in ms]); Xt=np.column_stack([T[m][0] for m in ms])
th=fit_simplex(Xv,yv); p=Xt@th[1:]+th[0]
cens=np.abs(yt-5.0)<1e-9
print('聚合预测: 删失行均值 %.3f | 非删失行均值 %.3f | 全体均值 %.3f'%(p[cens].mean(),p[~cens].mean(),p.mean()))
print('标签:     删失行均值 %.3f | 非删失行均值 %.3f'%(yt[cens].mean(),yt[~cens].mean()))
print('预测 sd %.3f | 标签 sd %.3f | 比值 %.3f'%(p.std(),yt.std(),p.std()/yt.std()))
print()
print('=== 分域 RMSE ===')
print('  全体      : %.4f'%rmse(p,yt))
print('  删失子集  : %.4f'%rmse(p[cens],yt[cens]))
print('  非删失子集: %.4f'%rmse(p[~cens],yt[~cens]))
# 各成员在非删失子集上的表现
print()
print('=== 非删失子集(仅 %d 行)上的成员排序 ==='%(~cens).sum())
rr=sorted([(rmse(Xt[~cens,i],yt[~cens]),ms[i]) for i in range(len(ms))])
for r,m in rr[:5]: print('   %-16s %.4f'%(m,r))
print('   --- 全体的成员排序 ---')
rr2=sorted([(rmse(Xt[:,i],yt),ms[i]) for i in range(len(ms))])
for r,m in rr2[:5]: print('   %-16s %.4f'%(m,r))
