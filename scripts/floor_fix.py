# -*- coding: utf-8 -*-
"""删失（检测下限）的三种修法 —— 检验能否成为原创算法层
关键事实：所有值 >= 5.0（min=5.0，次小 5.0044）→ 5.0 是报告值的下界，
弱结合被截断在 5.0。val 删失率 72.7% vs test 56.6% → 删失率本身发生位移。
无标签修法：
  (A) 删失率匹配重加权（用预测估计 test 删失率）
  (B) 方差恢复（把聚合预测的离散度拉回成员尺度）
  (C) 地板吸附（成员一致预测贴地板的行，吸附到地板）
"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
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
def detect_floor(y,minf=0.05):
    v,c=np.unique(np.round(y,6),return_counts=True); i=int(c.argmax())
    return (float(v[i]),float(c[i]/len(y))) if c[i]/len(y)>=minf else (None,0.0)
CELLS=[('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
       ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
print('%-16s %6s %6s %6s | %8s %8s %8s %8s'%('cell','floor','censVal','censTest','simplex','A:率匹配','B:方差恢复','C:吸附'))
rows=[]
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
    floor,cfv=detect_floor(yv)
    if floor is None:
        continue
    # 用预测估计 test 删失率：成员共识贴地板
    near=np.abs(Xt-floor).max(1)<0.75          # 所有成员都在地板附近
    cft=float(near.mean())
    cens_v=np.abs(yv-floor)<1e-6
    # 基线
    th=fit_simplex(Xv,yv); base=Xt@th[1:]+th[0]; base_rmse=rmse(base,yt)
    # (A) 删失率匹配：重加权使 val 的删失占比匹配 test
    w=np.where(cens_v, (1-cft)/max(1-cfv,1e-6), cft/max(cfv,1e-6))
    from scipy.optimize import lsq_linear
    A=np.column_stack([np.ones(len(yv)),Xv]); sw=np.sqrt(w/w.mean())
    r=lsq_linear((A*sw[:,None]).astype(np.float64), (yv*sw).astype(np.float64),
                 bounds=(np.array([-np.inf]+[0.0]*K), np.full(K+1,np.inf)), method='trf')
    thA=r.x[1:]/max(r.x[1:].sum(),1e-9); pA=Xt@thA+r.x[0]
    # (B) 方差恢复：把预测离散度拉回"成员共识"的尺度
    ref=Xt.mean(1); sd_ref=ref.std(); sd_p=base.std()
    pB=(base-base.mean())*(sd_ref/max(sd_p,1e-9))+base.mean() if sd_p>0 else base
    # (C) 地板吸附：成员共识贴地板的行 → 吸到地板
    pC=base.copy(); pC[near]=floor
    rows.append({'cell':cell,'floor':floor,'cens_val':cfv,'cens_test_est':cft,
                 'simplex':base_rmse,'A_rate_match':rmse(pA,yt),'B_var_restore':rmse(pB,yt),
                 'C_floor_snap':rmse(pC,yt),'true_cens_test':float((np.abs(yt-floor)<1e-6).mean())})
    print('%-16s %6.1f %5.0f%% %5.0f%% | %8.4f %8.4f %8.4f %8.4f'%(cell,floor,cfv*100,
        rows[-1]['true_cens_test']*100,base_rmse,rows[-1]['A_rate_match'],rows[-1]['B_var_restore'],rows[-1]['C_floor_snap']))
json.dump(rows,open('results/strengthen/floor_fix.json','w'),indent=1)
print()
for k,lab in [('A_rate_match','A 删失率匹配'),('B_var_restore','B 方差恢复'),('C_floor_snap','C 地板吸附')]:
    d=[r[k]-r['simplex'] for r in rows]
    print('  %-14s 中位 %+.4f  改好 %d/%d'%(lab,np.median(d),sum(1 for x in d if x<0),len(d)))
