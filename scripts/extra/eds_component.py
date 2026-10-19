# -*- coding: utf-8 -*-
"""端点分歧选择 (Endpoint-Divergence Selection, EDS) —— 原创组件 v2
前置组件：用成员跨端点的秩分歧度 δ_k 自动选出交互项集合 A。
对照：手工 A / 全量交互 / EDS 自动 A。全部 val 拟合，test 只评一次。"""
import os
import numpy as np, pandas as pd, os, sys, json
from scipy.optimize import lsq_linear
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES='results'; SD='sota_duel'
MEM=['f1','f2','f3','f4','f5','dtaom','e3mmdta','lightdta','esmmax']
STYPES=['IC50','Kd','Ki']
MAN='D:/protein_datasets/leap/frozen/battlefield_manifest.parquet'
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p,float)-np.asarray(y,float))**2)))
def build_side(df,sp):
    files={'f5':(os.path.join(RES,'deepdta_%s_pred.parquet'%sp),'deepdta'),
           'dtaom':(os.path.join(RES,'dtaom_%s_pred.parquet'%sp) if sp=='val' else os.path.join(SD,'dtaom','pred_test.parquet'),'dtaom' if sp=='val' else 'pred'),
           'e3mmdta':(os.path.join(SD,'e3mmdta','pred_%s.parquet'%sp),'pred'),
           'lightdta':(os.path.join(SD,'lightdta','pred_%s.parquet'%sp),'pred'),
           'esmmax':(os.path.join(SD,'esmmax','pred_%s.parquet'%sp),'pred')}
    for m,(f,col) in files.items():
        d=pd.read_parquet(f); c=col if col in d.columns else m
        if 'activity_id' not in d.columns: continue
        df=df.merge(d[['activity_id',c]].rename(columns={c:m}),on='activity_id',how='left')
    return df
man=pd.read_parquet(MAN,columns=['activity_id','split','tid','stype','pK'])
mp=pd.read_parquet(os.path.join(RES,'members_pred.parquet'))
def side(sp,ids):
    d=mp[mp['activity_id'].isin(set(ids))].copy()
    d=d.drop(columns=[c for c in ['stype','tid','pK'] if c in d.columns],errors='ignore')
    d=d.merge(man[['activity_id','tid','stype','pK']].rename(columns={'pK':'pK_m'}),on='activity_id',how='left')
    d['pK']=d['pK_m']; d=build_side(d,sp).dropna(subset=MEM).reset_index(drop=True)
    return d
val=side('val',man.loc[man.split=='val','activity_id'])
tst=side('test',pd.read_csv(os.path.join(SD,'data','test_chembl.csv'))['activity_id'].values)
yv=val['pK'].values.astype(float); yt=tst['pK'].values.astype(float)

def ranks_of(df):
    R={}
    for s in STYPES:
        m=df['stype'].values==s
        if m.sum()>=100: R[s]=[rmse(df.loc[m,k].values,df.loc[m,'pK'].values) for k in MEM]
    M=pd.DataFrame(R,index=MEM); return M.rank()
rk=ranks_of(val)
delta=rk.std(axis=1); dn=delta/delta.max()
thr=float(delta.median()); A_eds=[k for k in MEM if delta[k]>thr]
A_hand=['f1','f5','f2','f3']
print('δ_k（跨端点秩分歧）:')
for k in delta.sort_values(ascending=False).index:
    print('   %-10s δ=%.3f  秩IC50/Kd/Ki=%s %s'%(k,delta[k],list(rk.loc[k].values.astype(int)),
        '← EDS选中' if k in A_eds else ''))
print('\nEDS A = %s'%A_eds); print('手工 A = %s'%A_hand)

def fit(df,mems,A,use_type=True,use_inter=True):
    stv=df['stype'].values
    cols=[np.ones(len(df))]
    if use_type: cols.append(np.column_stack([(stv==s).astype(float) for s in STYPES if s!='IC50']))
    F=df[mems].values.astype(float); cols.append(F)
    n_free=1+(2 if use_type else 0); n_mem=len(mems); n_int=0
    if use_inter and A:
        E=np.column_stack([df[c].values.astype(float)*(stv==s).astype(float) for c in A for s in ['Kd','Ki']])
        cols.append(E); n_int=E.shape[1]
    Am=np.column_stack(cols)
    lb=np.array([-np.inf]*n_free+[0.0]*n_mem+[-np.inf]*n_int)
    r=lsq_linear(Am,df['pK'].values.astype(float),bounds=(lb,np.full(Am.shape[1],np.inf)),method='trf',tol=1e-10,max_iter=500)
    return r.x,n_free,n_mem,n_int
def pred(df,th,dims,A):
    n_free,n_mem,n_int=dims; stv=df['stype'].values
    cols=[np.ones(len(df)),np.column_stack([(stv==s).astype(float) for s in STYPES if s!='IC50']),df[MEM].values.astype(float)]
    if n_int>0 and A:
        cols.append(np.column_stack([df[c].values.astype(float)*(stv==s).astype(float) for c in A for s in ['Kd','Ki']]))
    return np.column_stack(cols)@th

print('\n=== 对照（test RMSE）===')
out={}
cfgs=[('L1 无交互',[]),('手工 A={f1,f5,f2,f3}',A_hand),('EDS A (自动)',A_eds),('全量 9 成员',MEM)]
for name,A in cfgs:
    th,nf,nm,ni=fit(val,MEM,A); pt=pred(tst,th,(nf,nm,ni),A)
    out[name]=rmse(pt,yt)
    print('  %-24s %.5f'%(name,out[name]))
th0,nf0,nm0,ni0=fit(val,MEM,[]); base=rmse(pred(tst,th0,(nf0,nm0,ni0),[]),yt)
print('\n相对无交互的增益:')
for k,v in out.items():
    if k!='L1 无交互': print('  %-24s %+.5f'%(k,v-base))
json.dump({'delta':{k:float(delta[k]) for k in MEM},
           'ranks':{k:[int(x) for x in rk.loc[k].values] for k in MEM},
           'rank_endpoints':list(rk.columns),
           'A_eds':A_eds,'A_hand':A_hand,'threshold':thr,
           'test_rmse':out,'base_no_interaction':base,
           'gain_vs_base':{k:v-base for k,v in out.items() if k!='L1 无交互'}},
          open('results/strengthen/eds_component.json','w'),indent=1)
print('\nWROTE results/strengthen/eds_component.json')
