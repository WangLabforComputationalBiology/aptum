# -*- coding: utf-8 -*-
"""原创层候选：交叉拟合反稀释收缩 (Cross-Fitted Anti-Dilution Shrinkage, CADS)
动机：论文自己诊断出"稀释"机制（kiba_cold：权重正确但被 0.589 的弱成员稀释）。
规则：p = λ·p_anchor + (1−λ)·p_agg，anchor = val 表现最好的单成员，
      λ 由 val 内部 5 折交叉拟合选定（无标签、无 test 信息）。
在 14 格上检验：能否把三红格变正 + 是否全局安全。"""
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
M={}
for r in json.load(open('results/matrix_4x5.json'))['rows']: M[(r['dataset'],r['protocol'])]=r
CELLS=[('chembl_temporal','ChEMBL 战场','时间'),('chembl_random','ChEMBL 战场','随机'),
 ('chembl_scaffold','ChEMBL 战场','scaffold'),('chembl_cold','ChEMBL 战场','冷启动'),
 ('bindingdb_temporal','BindingDB','时间'),('bindingdb_random','BindingDB','随机'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','冷启动'),
 ('davis_official','Davis','随机'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','冷启动'),
 ('kiba_official','KIBA','随机'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','冷启动')]
LAMS=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
def fitter(d):  # d='simplex' or 'nnls'
    return fit_simplex if d=='simplex' else fit_nnls
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
    row=M.get((ds,pr))
    if row is None or not row.get('opponent'): continue
    tr=np.array([rmse(Xt[:,i],yt) for i in range(K)])
    # val 内部 5 折交叉拟合选 λ
    rng=np.random.default_rng(0); n=len(yv); idx=rng.permutation(n); folds=np.array_split(idx,5)
    cv=np.zeros(len(LAMS))
    for f in folds:
        trn=np.setdiff1d(np.arange(n),f)
        t_=fit_nnls(Xv[trn],yv[trn])         # full aggregate on 4 folds
        p_agg=Xv[f]@t_[1:]+t_[0]
        vr=[rmse(Xv[trn,i],yv[trn]) for i in range(K)]
        a=int(np.argmin(vr))                  # anchor selected on 4 folds
        for j,l in enumerate(LAMS):
            cv[j]+=np.sum((l*Xv[f,a]+(1-l)*p_agg-yv[f])**2)
    lam=LAMS[int(np.argmin(cv))]
    # 用全 val 重拟合 + 全 val 选 anchor
    t_full=fit_nnls(Xv,yv); vr_full=[rmse(Xv[:,i],yv) for i in range(K)]
    a_full=int(np.argmin(vr_full))
    p_agg_t=Xt@t_full[1:]+t_full[0]
    p_cads=lam*Xt[:,a_full]+(1-lam)*p_agg_t
    # simplex 版本
    ts_full=fit_simplex(Xv,yv); p_sim=Xt@ts_full[1:]+ts_full[0]
    p_cads_s=lam*Xt[:,a_full]+(1-lam)*p_sim
    r={'cell':cell,'K':K,'lambda':lam,'anchor':ms[a_full],
       'strict_opp':float(tr.min()),'fair_opp':float(tr[int(np.argmin([rmse(Xv[:,i],yv) for i in range(K)]))]),
       'matrix_ours':row['ours']['rmse'],
       'deployed_nnls':rmse(p_agg_t,yt),'cads_nnls':rmse(p_cads,yt),
       'deployed_simplex':rmse(p_sim,yt),'cads_simplex':rmse(p_cads_s,yt)}
    r['strict_win']={'cads_nnls':bool(r['cads_nnls']<tr.min()),'cads_simplex':bool(r['cads_simplex']<tr.min())}
    rows.append(r)
    print('%-18s λ=%.1f anchor=%-14s | strict %.4f | sim %.4f→CADS %.4f %s | nnls %.4f→CADS %.4f %s'%(
        cell,lam,ms[a_full],tr.min(),r['deployed_simplex'],r['cads_simplex'],
        '✅' if r['cads_simplex']<tr.min() else '  ',r['deployed_nnls'],r['cads_nnls'],
        '✅' if r['cads_nnls']<tr.min() else '  '))
json.dump(rows,open('results/strengthen/antisilution.json','w'),indent=1)
print()
print('=== 汇总 ===')
for k,lab in [('cads_simplex','CADS(simplex)'),('cads_nnls','CADS(nnls)')]:
    w=sum(1 for r in rows if r['strict_win'][k]); print('  %-16s 严格赢(对test-oracle) %2d/%d'%(lab,w,len(rows)))
print('=== vs 已部署 ===')
for k,dep in [('cads_simplex','deployed_simplex'),('cads_nnls','deployed_nnls')]:
    d=[r[k]-r[dep] for r in rows]
    print('  %-16s 中位差 %+.4f  改好 %d/%d'%(k,np.median(d),sum(1 for x in d if x<0),len(d)))
