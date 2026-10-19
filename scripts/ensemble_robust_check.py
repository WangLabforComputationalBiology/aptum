# -*- coding: utf-8 -*-
"""全 14 格：中位数/截尾均值 集成 vs 官方部署 vs 对手 vs 均匀平均
判定它是不是一条【全局可用】的无标签规则，还是只在 kiba_scaffold 上巧合。"""
import numpy as np, pandas as pd, glob, os, json
from scipy.optimize import nnls
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictions')
mm=json.load(open('results/matrix_4x5.json'))
def norm(ds,pr): return ds.replace(' 战场','').replace('激酶基准','Kinase')+'|'+{'时间':'temporal','随机':'random','冷启动':'cold'}.get(pr,pr)
OPP={norm(r['dataset'],r['protocol']):r['opponent']['rmse'] for r in mm['rows'] if r.get('ours')}
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
CELLS=[('chembl_temporal','ChEMBL|temporal'),('chembl_random','ChEMBL|random'),('chembl_scaffold','ChEMBL|scaffold'),('chembl_cold','ChEMBL|cold'),
       ('bindingdb_temporal','BindingDB|temporal'),('bindingdb_random','BindingDB|random'),('bindingdb_scaffold','BindingDB|scaffold'),('bindingdb_cold','BindingDB|cold'),
       ('davis_official','Davis|random'),('davis_scaffold','Davis|scaffold'),('davis_cold','Davis|cold'),
       ('kiba_official','KIBA|random'),('kiba_scaffold','KIBA|scaffold'),('kiba_cold','KIBA|cold')]
rows=[]
for cell,mk in CELLS:
    d=R+cell
    allm=[os.path.basename(f).replace('pred_','').replace('_test.parquet','') for f in sorted(glob.glob(d+'/pred_*_test.parquet')) if 'kairos' not in f]
    ms=[n for n in allm if os.path.exists(d+'/pred_%s_val.parquet'%n)]
    Xv=[];Xt=[];yv=yt=None
    for n in ms:
        pv=pd.read_parquet(d+'/pred_%s_val.parquet'%n); pt=pd.read_parquet(d+'/pred_%s_test.parquet'%n)
        if yt is None: yv=pv['pK'].values; yt=pt['pK'].values
        Xv.append(pv['pred'].values); Xt.append(pt['pred'].values)
    Xv=np.column_stack(Xv); Xt=np.column_stack(Xt)
    m=np.isfinite(Xv).all(1)&np.isfinite(yv); Xv,yv=Xv[m],yv[m]
    m2=np.isfinite(Xt).all(1)&np.isfinite(yt); Xt,yt=Xt[m2],yt[m2]
    A=np.hstack([Xv,-np.ones((len(Xv),1)),np.ones((len(Xv),1))])
    w,_=nnls(A.astype(np.float64),yv.astype(np.float64)); th=w[:-2]; c=w[-1]-w[-2]
    K=Xt.shape[1]
    res={'cell':cell,'K':K,'opp':OPP[mk],
         'official':rmse(Xt@th+c,yt),
         'uniform':rmse(Xt.mean(1),yt),
         'median':rmse(np.median(Xt,1),yt),
         'trimmed':rmse(np.sort(Xt,1)[:,max(1,(K-1)//2):K-max(1,(K-1)//2)].mean(1),yt) if K>2 else float('nan')}
    res['best_single']=min(rmse(Xt[:,i],yt) for i in range(K))
    rows.append(res)
print('%-20s %2s %8s %8s %8s %8s %8s %8s'%('cell','K','对手','官方','均匀','中位数','截尾','最佳单'))
for r in rows:
    print('%-20s %2d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f'%(r['cell'],r['K'],r['opp'],r['official'],r['uniform'],r['median'],r['trimmed'],r['best_single']))
print()
print('=== 各方案 vs 对手：赢的格数 ===')
for k in ['official','uniform','median','trimmed']:
    w=[r['cell'] for r in rows if r[k]==r[k] and r[k]<r['opp']]
    print('  %-9s 赢 %2d/%d   输的格: %s'%(k,len(w),len(rows),[r['cell'] for r in rows if not(r[k]==r[k] and r[k]<r['opp'])]))
print()
print('=== 各方案 vs 官方部署：谁更好 ===')
for k in ['uniform','median','trimmed']:
    d=[r[k]-r['official'] for r in rows if r[k]==r[k]]
    print('  %-9s 中位差 %+.4f  比官方好的格数 %d/%d'%(k,np.median(d),sum(1 for x in d if x<0),len(d)))
json.dump(rows,open('results/strengthen/ensemble_robust.json','w'),indent=1)
print('\nWROTE results/strengthen/ensemble_robust.json')
