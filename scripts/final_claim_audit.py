# -*- coding: utf-8 -*-
"""最终 claim 审计：Kairos(cv5) vs 全部格内方法（含单成员、均匀、中位数、对手）
两种口径：strict(test-oracle 对手) / 全池成员对比
输出每格的 排名第一吗 + 增益能力（相对最佳单成员的降幅%）"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
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
def cv(X,y,fit,nf=5,seed=0):
    n=len(y); rng=np.random.default_rng(seed); idx=rng.permutation(n); p=np.zeros(n)
    for f in np.array_split(idx,nf):
        t=np.setdiff1d(np.arange(n),f); th=fit(X[t],y[t]); p[f]=X[f]@th[1:]+th[0]
    return p
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
print('%-18s %3s | %8s %8s %8s | %5s %5s | %6s %6s'%('cell','K','最佳单成员','对手(ora)','Kairos-cv5','#1成员','#1全池','增益%','vs对手%'))
rows=[];n_mem_first=0;n_all_first=0;n_cells=0
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
    tr=np.array([rmse(Xt[:,i],yt) for i in range(K)])
    best_single=float(tr.min()); opp=row['opponent']['rmse']
    uni=rmse(Xt.mean(1),yt); med=rmse(np.median(Xt,1),yt)
    kai=rmse(cv(Xt,yt,fit_simplex),yt)
    # 排名：Kairos 在 {所有单成员 + 对手 + 均匀 + 中位数} 里的位次
    contenders=dict({ms[i]:tr[i] for i in range(K)},uniform=uni,median=med)
    rank_mem=1+sum(1 for k,v in {ms[i]:tr[i] for i in range(K)}.items() if v<kai)
    rank_all=1+sum(1 for k,v in contenders.items() if v<kai)
    gain=(best_single-kai)/best_single*100
    vs_opp=(opp-kai)/opp*100
    n_cells+=1; n_mem_first+=(rank_mem==1); n_all_first+=(rank_all==1)
    r={'cell':cell,'K':K,'best_single':best_single,'opponent':opp,'kairos_cv5':kai,
       'uniform':uni,'median':med,'rank_vs_members':rank_mem,'rank_vs_all':rank_all,
       'gain_pct':gain,'vs_opponent_pct':vs_opp}
    rows.append(r)
    print('%-18s %3d | %8.4f %8.4f %8.4f | %5d %5d | %+6.2f %+6.2f'%(cell,K,best_single,opp,kai,
        rank_mem,rank_all,gain,vs_opp))
json.dump(rows,open('results/strengthen/final_claim_audit.json','w'),indent=1)
print()
print('=== 结论 ===')
print('  格数 %d'%n_cells)
print('  Kairos(cv5) 打败【所有单成员】的格: %d/%d'%(n_mem_first,n_cells))
print('  Kairos(cv5) 打败【所有单成员+均匀+中位数】的格: %d/%d'%(n_all_first,n_cells))
g=[r['gain_pct'] for r in rows]
print('  相对最佳单成员的降幅: 中位 %+.2f%%  范围 %+.2f%%~%+.2f%%'%(np.median(g),min(g),max(g)))
print('  负增益(不如最佳单成员)的格: %s'%[r['cell'] for r in rows if r['gain_pct']<0])
