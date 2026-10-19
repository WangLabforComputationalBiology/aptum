# -*- coding: utf-8 -*-
"""KIBA·scaffold 为什么是唯一输给均匀平均的格：成分、val→test 可迁移性、各种修法"""
import numpy as np, pandas as pd, glob, os, json
from scipy.optimize import nnls
from scipy import stats
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictions')
CELL='kiba_scaffold'
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

# ---- 1. 成员预测与 val/test 成分 ----
fs=[f for f in sorted(glob.glob(R+CELL+'/pred_*_val.parquet')) if 'kairos' not in f]
mem=[os.path.basename(f).replace('pred_','').replace('_val.parquet','') for f in fs]
V={}; T={}
for n in mem:
    pv=pd.read_parquet(R+CELL+'/pred_%s_val.parquet'%n); pt=pd.read_parquet(R+CELL+'/pred_%s_test.parquet'%n)
    V[n]=(pv['pred'].values, pv['pK'].values); T[n]=(pt['pred'].values, pt['pK'].values)
yv=V[mem[0]][1]; yt=T[mem[0]][1]
Xv=np.column_stack([V[n][0] for n in mem]); Xt=np.column_stack([T[n][0] for n in mem])
print('=== 0. 规模 ===')
print('  val  %d 行 | test %d 行 | 池 K=%d'%(len(yv),len(yt),len(mem)))
print('  val y: mean %.3f sd %.3f  | test y: mean %.3f sd %.3f'%(yv.mean(),yv.std(),yt.mean(),yt.std()))

# 成分（从 prepared parquet 拿 scaffold/target）
df=pd.read_parquet('data/splits/kiba_prepared.parquet')
print('  prepared 列:', [c for c in df.columns])
print('=== 1. 划分成分（scaffold 分组）===')
if 'split_scaffold' in df.columns:
    for sp in ['train','val','test']:
        d=df[df.split_scaffold==sp]
        print('  %-5s 行 %6d | 唯一smiles %5d | 唯一target %4d | 唯一scaffold %5d'%(
            sp,len(d),d['smiles'].nunique() if 'smiles' in d else -1,
            d['target_key'].nunique() if 'target_key' in d else -1,
            d['scaffold'].nunique() if 'scaffold' in d else -1))

# ---- 2. val 排名 vs test 排名（各格的对比基准） ----
print('=== 2. val→test 可迁移性 ===')
vv={n:rmse(V[n][0],yv) for n in mem}; tt={n:rmse(T[n][0],yt) for n in mem}
rho=stats.spearmanr([vv[n] for n in mem],[tt[n] for n in mem]).correlation
print('  本格 Spearman(val,test) = %.3f'%rho)
order_v=sorted(mem,key=lambda n:vv[n]); order_t=sorted(mem,key=lambda n:tt[n])
print('  %-16s %8s %4s %8s %4s'%('member','valRMSE','v#','testRMSE','t#'))
for n in order_v:
    print('  %-16s %8.4f %4d %8.4f %4d'%(n,vv[n],order_v.index(n)+1,tt[n],order_t.index(n)+1))

def fit_nnls(X,y):
    A=np.hstack([X,-np.ones((len(X),1)),np.ones((len(X),1))])
    w,_=nnls(A.astype(np.float64),y.astype(np.float64)); return w[:-2], w[-1]-w[-2]
th,c=fit_nnls(Xv,yv)
print('=== 3. 各口径 test RMSE ===')
cands={
 'val NNLS(部署)': rmse(Xt@th+c,yt),
 '均匀平均': rmse(Xt.mean(1),yt),
 '中位数集成': rmse(np.median(Xt,1),yt),
 '截尾均值': rmse(np.sort(Xt,1)[:,1:-1].mean(1),yt),
}
# val 选优单成员 / val 最差剔除
best_v=order_v[0]; worst_v=order_v[-1]
cands['val 最优单成员 (%s)'%best_v]=rmse(Xt[:,mem.index(best_v)],yt)
keep=[i for i,n in enumerate(mem) if n!=worst_v]
th2,c2=fit_nnls(Xv[:,keep],yv)
cands['去掉 val 最差 (%s) 重拟合'%worst_v]=rmse(Xt[:,keep]@th2+c2,yt)
# 收缩
for lam in [0.3,0.5,0.7]:
    w=lam*th+(1-lam)/len(mem); cc=lam*c
    cands['θ 向均匀收缩 λ=%.1f'%lam]=rmse(Xt@w+cc,yt)
# oracle
th_o,c_o=fit_nnls(Xt,yt); cands['oracle(test拟合)']=rmse(Xt@th_o+c_o,yt)
for k,v in sorted(cands.items(),key=lambda x:x[1]):
    print('  %-34s %.4f'%(k,v))
print('  对手 f1full_xgb %.4f'%tt['f1full_xgb'])

# ---- 4. θ 分布：过度集中？ ----
print('=== 4. val NNLS 的 θ（前 5）===')
for i in np.argsort(-th)[:5]:
    print('  %-16s θ=%.4f  val#%d test#%d'%(mem[i],th[i],order_v.index(mem[i])+1,order_t.index(mem[i])+1))
print('  Σθ=%.4f 截距=%+.4f  有效成员数(1/Σθ²)=%.1f'%(th.sum(),c,1/np.sum(th**2)))
