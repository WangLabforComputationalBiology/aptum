# -*- coding: utf-8 -*-
"""汇总型提升指标：几种口径全算出来，供论文选用"""
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
cv5=json.load(open('results/strengthen/cv5_ci.json'))
# 基线：uniform / best_single
UB=json.load(open('results/exp_gap/matrix_uniform_baseline.json'))['cells']
Q01=json.load(open('results/exp_gap/q/q01.json'))
CELLS=[('chembl_temporal','ChEMBL','temporal'),('chembl_random','ChEMBL','random'),
 ('chembl_scaffold','ChEMBL','scaffold'),('chembl_cold','ChEMBL','cold'),
 ('bindingdb_temporal','BindingDB','temporal'),('bindingdb_random','BindingDB','random'),
 ('bindingdb_scaffold','BindingDB','scaffold'),('bindingdb_cold','BindingDB','cold'),
 ('davis_official','Davis','random'),('davis_scaffold','Davis','scaffold'),('davis_cold','Davis','cold'),
 ('kiba_official','KIBA','random'),('kiba_scaffold','KIBA','scaffold'),('kiba_cold','KIBA','cold')]
rows=[]
for cell,ds,pr in CELLS:
    if not os.path.isdir('results/p2/%s'%cell): continue
    ms=memb(cell)
    y=pd.read_parquet('results/p2/%s/pred_%s_test.parquet'%(cell,ms[0]))['pK'].values.astype(float)
    bs=min(rmse(pd.read_parquet('results/p2/%s/pred_%s_test.parquet'%(cell,m))['pred'].values.astype(float),y) for m in ms)
    if cell in UB:
        uni=float(UB[cell]['uniform']) if not isinstance(UB[cell]['uniform'],dict) else UB[cell]['uniform']['rmse']
    else:
        uni=float(Q01[cell]['uniform'])
    ours=float(cv5[cell]['rmse'])
    n=int(cv5[cell]['n'])
    rows.append({'cell':cell,'ds':ds,'ours':ours,'best_single':bs,'uniform':uni,'n':n,
                 'gain_bs':100*(bs-ours)/bs,'gain_uni':100*(uni-ours)/uni,
                 'gain_min':100*(min(bs,uni)-ours)/min(bs,uni)})
df=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(df[['cell','ours','best_single','uniform','gain_bs','gain_uni','gain_min']].to_string(index=False,float_format=lambda x:'%.3f'%x))
print()
print('='*70)
print('【口径 A】逐格百分比的平均（每格等权）')
print('   vs 最佳单成员    : 平均 %+.2f%%  中位 %+.2f%%'%(df.gain_bs.mean(),df.gain_bs.median()))
print('   vs 均匀平均      : 平均 %+.2f%%  中位 %+.2f%%'%(df.gain_uni.mean(),df.gain_uni.median()))
print('   vs 更强基线      : 平均 %+.2f%%  中位 %+.2f%%'%(df.gain_min.mean(),df.gain_min.median()))
print()
print('【口径 B】池化 RMSE（按测试行数加权，更像"总体误差"）')
w=df.n.values
for lab,col in [('最佳单成员','best_single'),('均匀平均','uniform')]:
    base_pooled=np.sqrt(np.average(df[col].values**2,weights=w))
    ours_pooled=np.sqrt(np.average(df.ours.values**2,weights=w))
    print('   vs %-8s 池化 %.4f -> %.4f  提升 %+.2f%%'%(lab,base_pooled,ours_pooled,100*(base_pooled-ours_pooled)/base_pooled))
print()
print('【口径 C】逐格 RMSE 差值平均（pK 单位，最保守）')
print('   vs 最佳单成员: 平均 %.4f pK'%(df.best_single-df.ours).mean())
print('   vs 均匀平均  : 平均 %.4f pK'%(df.uniform-df.ours).mean())
print()
print('【口径 D】只算"有公开对手"的格（Davis/KIBA 随机 + 复现的第三方）')
print('   第三方复现值: Davis 0.5224/0.5272/0.6468, KIBA 0.4609/0.4831')
print('   Kairos Davis 0.5011 -> vs GraphDTA 0.5224 = %+.2f%%'%(100*(0.5224-0.5011)/0.5224))
print('   Kairos KIBA  0.4151 -> vs GraphDTA 0.4609 = %+.2f%%'%(100*(0.4609-0.4151)/0.4609))
json.dump(rows,open('results/strengthen/aggregate_gain.json','w'),indent=1)
