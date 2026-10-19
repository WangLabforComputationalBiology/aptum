# -*- coding: utf-8 -*-
"""算出每个端点的 θ 权重（旗舰格），供单纯形几何图使用。"""
import os
import numpy as np, pandas as pd, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_simplex, fit_nnls
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = 'results'; SD = 'sota_duel'
MEM = ['f1','f2','f3','f4','f5','dtaom','e3mmdta','lightdta','esmmax']
STYPES = ['IC50','Kd','Ki']
MAN = 'D:/protein_datasets/leap/frozen/battlefield_manifest.parquet'

def build_side(df, sp):
    files = {'f5':(os.path.join(RES,'deepdta_%s_pred.parquet'%sp),'deepdta'),
             'dtaom':(os.path.join(RES,'dtaom_%s_pred.parquet'%sp) if sp=='val'
                      else os.path.join(SD,'dtaom','pred_test.parquet'),
                      'dtaom' if sp=='val' else 'pred'),
             'e3mmdta':(os.path.join(SD,'e3mmdta','pred_%s.parquet'%sp),'pred'),
             'lightdta':(os.path.join(SD,'lightdta','pred_%s.parquet'%sp),'pred'),
             'esmmax':(os.path.join(SD,'esmmax','pred_%s.parquet'%sp),'pred')}
    for m,(f,col) in files.items():
        d = pd.read_parquet(f); c = col if col in d.columns else m
        if 'activity_id' not in d.columns: continue
        df = df.merge(d[['activity_id',c]].rename(columns={c:m}), on='activity_id', how='left')
    return df

man = pd.read_parquet(MAN, columns=['activity_id','split','stype','pK'])
mp = pd.read_parquet(os.path.join(RES,'members_pred.parquet'))
val = mp[mp['activity_id'].isin(set(man.loc[man.split=='val','activity_id']))].copy()
val = val.drop(columns=[c for c in ['stype','tid','pK'] if c in val.columns], errors='ignore')
val = val.merge(man[['activity_id','stype','pK']].rename(columns={'pK':'pK_m'}),
                on='activity_id', how='left')
val['pK'] = val['pK_m']
val = build_side(val,'val').dropna(subset=MEM).reset_index(drop=True)
print('n_val =', len(val))

out = {'members': MEM, 'endpoints': {}, 'pool': {}}
Xall = val[MEM].values.astype(float); yall = val['pK'].values.astype(float)
th = fit_simplex(Xall, yall)
out['pool'] = {'theta': [float(t) for t in th[1:]], 'intercept': float(th[0])}
print('pool simplex theta:', {MEM[i]: round(th[1+i],3) for i in range(len(MEM))})

for s in STYPES:
    m = val['stype'].values == s
    if m.sum() < 100: continue
    X = val.loc[m, MEM].values.astype(float); y = val.loc[m,'pK'].values.astype(float)
    t = fit_simplex(X, y)
    rm = [float(np.sqrt(np.mean((X[:,i]-y)**2))) for i in range(len(MEM))]
    out['endpoints'][s] = {'n': int(m.sum()), 'theta': [float(x) for x in t[1:]],
                           'intercept': float(t[0]),
                           'member_rmse': rm,
                           'rank': [int(r) for r in pd.Series(rm).rank().values],
                           'mean_pK': float(y.mean())}
    print('%-5s n=%6d  theta(top3)=%s  mean pK=%.3f' % (
        s, m.sum(),
        sorted([(round(t[1+i],3), MEM[i]) for i in range(len(MEM))], reverse=True)[:3],
        y.mean()))

json.dump(out, open('results/strengthen/per_endpoint_theta.json','w'), indent=1)
print('\nWROTE results/strengthen/per_endpoint_theta.json')