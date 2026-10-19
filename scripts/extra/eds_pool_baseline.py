# -*- coding: utf-8 -*-
"""在 EDS 的同一份 val/test 数据上，补算池内最佳单成员，供 Figure 2 (fig:eds) 使用。
口径与 eds_component.py 完全一致（同划分、同成员、同 rmse 定义）。
"""
import os
import numpy as np, pandas as pd, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = 'results'; SD = 'sota_duel'
MEM = ['f1','f2','f3','f4','f5','dtaom','e3mmdta','lightdta','esmmax']
MAN = 'D:/protein_datasets/leap/frozen/battlefield_manifest.parquet'

def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))

def build_side(df, sp):
    files = {'f5': (os.path.join(RES, 'deepdta_%s_pred.parquet' % sp), 'deepdta'),
             'dtaom': (os.path.join(RES, 'dtaom_%s_pred.parquet' % sp) if sp == 'val'
                       else os.path.join(SD, 'dtaom', 'pred_test.parquet'),
                       'dtaom' if sp == 'val' else 'pred'),
             'e3mmdta': (os.path.join(SD, 'e3mmdta', 'pred_%s.parquet' % sp), 'pred'),
             'lightdta': (os.path.join(SD, 'lightdta', 'pred_%s.parquet' % sp), 'pred'),
             'esmmax': (os.path.join(SD, 'esmmax', 'pred_%s.parquet' % sp), 'pred')}
    for m, (f, col) in files.items():
        d = pd.read_parquet(f); c = col if col in d.columns else m
        if 'activity_id' not in d.columns: continue
        df = df.merge(d[['activity_id', c]].rename(columns={c: m}),
                      on='activity_id', how='left')
    return df

man = pd.read_parquet(MAN, columns=['activity_id', 'split', 'stype', 'pK'])
mp = pd.read_parquet(os.path.join(RES, 'members_pred.parquet'))

def side(sp, ids):
    d = mp[mp['activity_id'].isin(set(ids))].copy()
    d = d.drop(columns=[c for c in ['stype', 'tid', 'pK'] if c in d.columns], errors='ignore')
    d = d.merge(man[['activity_id', 'stype', 'pK']].rename(columns={'pK': 'pK_m'}),
                on='activity_id', how='left')
    d['pK'] = d['pK_m']
    d = build_side(d, sp).dropna(subset=MEM).reset_index(drop=True)
    return d

tst = side('test', pd.read_csv(os.path.join(SD, 'data', 'test_chembl.csv'))['activity_id'].values)
yv = tst['pK'].values.astype(float)
member_test = {m: rmse(tst[m].values.astype(float), yv) for m in MEM}
best_m = min(member_test, key=member_test.get)

out = {'n_test': int(len(tst)),
       'member_rmse': member_test,
       'best_single_member': best_m,
       'best_single_rmse': member_test[best_m],
       'uniform_avg_rmse': rmse(tst[MEM].values.astype(float).mean(1), yv)}
print('n_test =', out['n_test'])
print('pool member test RMSE (sorted):')
for m, v in sorted(member_test.items(), key=lambda kv: kv[1]):
    print('   %-10s %.5f %s' % (m, v, '<- best' if m == best_m else ''))
print('uniform average : %.5f' % out['uniform_avg_rmse'])
print('best single (%s): %.5f' % (best_m, out['best_single_rmse']))
json.dump(out, open('results/strengthen/eds_pool_baseline.json', 'w'), indent=1)
print('\nWROTE results/strengthen/eds_pool_baseline.json')