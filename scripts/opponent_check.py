# -*- coding: utf-8 -*-
"""验证：论文的"对手"是 val 选还是 test 选（这决定三格的性质）"""
import os
import numpy as np, pandas as pd, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
print('%-18s %-14s %9s %9s | %-10s %-10s %s'%('cell','论文对手','其RMSE','','test选成员','val选成员','判定'))
n_test=n_val=0
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
    mv=np.isfinite(np.column_stack([V[m][0] for m in ms])).all(1)&np.isfinite(yv)
    mt=np.isfinite(np.column_stack([T[m][0] for m in ms])).all(1)&np.isfinite(yt)
    vr=np.array([rmse(np.column_stack([V[m][0] for m in ms])[mv,i],yv[mv]) for i in range(K)])
    tr=np.array([rmse(np.column_stack([T[m][0] for m in ms])[mt,i],yt[mt]) for i in range(K)])
    tb,vb=int(tr.argmin()),int(vr.argmin())
    row=M[(ds,pr)]; oppname=row['opponent']['name']; opprm=row['opponent']['rmse']
    is_test = (ms[tb]==oppname) and abs(tr[tb]-opprm)<1e-6
    is_val  = (ms[vb]==oppname) and abs(tr[vb]-opprm)<1e-6
    n_test+=is_test; n_val+=is_val
    print('%-18s %-14s %9.4f %9s | %-10s %-10s %s'%(cell,oppname,opprm,'',
        ms[tb],ms[vb],'test-oracle' if is_test else ('val选' if is_val else '??')))
print()
print('对手 = test-oracle 的格数: %d/%d'%(n_test,len(CELLS)))
print('对手 = val 选出的格数: %d/%d'%(n_val,len(CELLS)))
