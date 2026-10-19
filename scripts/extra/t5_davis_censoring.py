# -*- coding: utf-8 -*-
"""t5-(1) Davis 5.0 删失审计 + 常数预测器对照 + 6 化合物聚类 bootstrap。
纯 CPU，只读冻结产物，不重训成员。
输出 D:/protein_datasets/midas/results/strengthen/t5/davis_censoring.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

T0 = time.time()


def prog(m):
    print('[%6.1fs] %s' % (time.time() - T0, m), flush=True)


prog('start')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_nnls, fit_simplex, apply_th
prog('kairos_rule imported')

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(MID, 'results', 'strengthen', 't5')
os.makedirs(OUT, exist_ok=True)
CENSOR = 5.0
MATRIX = json.load(open(os.path.join(MID, 'results', 'matrix_4x5.json'), encoding='utf-8'))
KR = json.load(open(os.path.join(MID, 'results', 'kairos_theta_rule.json'), encoding='utf-8'))
ROWW = {r['dataset'] + '|' + r['protocol']: r for r in MATRIX['rows']}


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def mse(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# ---------------------------------------------------------------- 1. 删失占比
dp = pd.read_parquet(os.path.join(MID, 'data', 'splits', 'davis_prepared.parquet'))
censor = {
    'definition': 'Davis pK == 5.0 (Kd = 10000 nM, assay upper limit / right-censored at zero-affinity ceiling)',
    'n_rows_total': int(len(dp)),
    'n_censored_total': int((dp.pK == CENSOR).sum()),
    'frac_censored_total': round(float((dp.pK == CENSOR).mean()), 6),
    'pK_min': float(dp.pK.min()), 'pK_max': float(dp.pK.max()), 'pK_mean': float(dp.pK.mean()),
    'second_lowest_pK': float(np.sort(dp.pK.unique())[1]),
    'frac_censored_at_second_lowest_and_below': round(float((dp.pK <= np.sort(dp.pK.unique())[1]).mean()), 6),
    'per_split': {},
}
for col, cell in [('split_scaffold', 'davis_scaffold'), ('split_cold', 'davis_cold')]:
    block = {}
    for sp in ['train', 'val', 'test']:
        s = dp[dp[col] == sp]
        block[sp] = {
            'n_rows': int(len(s)),
            'n_censored': int((s.pK == CENSOR).sum()),
            'frac_censored': round(float((s.pK == CENSOR).mean()), 6),
            'pK_mean': round(float(s.pK.mean()), 6),
            'n_unique_smiles': int(s.smiles.nunique()),
            'n_unique_target_key': int(s.target_key.nunique()),
            'n_unique_scaffold': int(s.scaffold.nunique()),
        }
    censor['per_split'][col] = block
# 官方 setting1 (davis_preds) 的删失占比
dpr = pd.read_parquet(os.path.join(MID, 'sota_duel', 'ext', 'davis_preds.parquet'))
prog('davis_prepared + davis_preds loaded')
censor['official_setting1'] = {
    'definition': 'results/p2/davis_official 使用的官方 setting1 单划分（sota_duel/ext/davis_preds.parquet）',
    'per_split': {sp: {'n_rows': int((dpr.split == sp).sum()),
                       'n_censored': int((dpr[dpr.split == sp].pK == CENSOR).sum()),
                       'frac_censored': round(float((dpr[dpr.split == sp].pK == CENSOR).mean()), 6)}
                  for sp in ['train', 'test']},
}

# ------------------------------------------------- 2. 含/剔 5.0 行 的 RMSE 重建
MEMBERS = KR['cells']['davis_scaffold']['members']


def load_P(cell, members, tag):
    Ps, y = [], None
    for n in members:
        d = pd.read_parquet(os.path.join(MID, 'results', 'p2', cell, 'pred_%s_%s.parquet' % (n, tag)))
        if y is None:
            y = d['pK'].values
        else:
            assert np.allclose(y, d['pK'].values), (cell, n, tag)
        Ps.append(d['pred'].values)
    return np.column_stack(Ps), np.asarray(y, float)


recon = {}
for cell, mkey, ckey in [('davis_scaffold', 'Davis|scaffold', 'split_scaffold'),
                         ('davis_cold', 'Davis|冷启动', 'split_cold')]:
    row = ROWW[mkey]
    th_official = np.asarray(row['theta'], float)
    opp_name = row['opponent']['name']
    members = KR['cells'][cell]['members']
    Pv, yv = load_P(cell, members, 'val')
    Pt, yt = load_P(cell, members, 'test')
    prog('%s loaded Pv%r Pt%r' % (cell, Pv.shape, Pt.shape))
    assert members == KR['cells'][cell]['members']
    # 行对齐到 prepared 文件（按 pK 逐元素）
    sub_t = dp[dp[ckey] == 'test'].reset_index(drop=True)
    assert np.allclose(sub_t['pK'].values, yt)
    sub_v = dp[dp[ckey] == 'val'].reset_index(drop=True)
    assert np.allclose(sub_v['pK'].values, yv)
    mask_t = (yt == CENSOR)
    mask_v = (yv == CENSOR)

    p_official = apply_th(th_official, Pt)
    p_opp = Pt[:, members.index(opp_name)]

    # (a) 固定 θ（落盘 official θ），在 全/剔 5.0 子集上评估
    fixed = {
        'theta_source': 'results/matrix_4x5.json 该格 theta（val 拟合，simplex/nnls 由 θ 规则选）',
        'theta': [float(x) for x in th_official],
        'all_rows': {'n': int(len(yt)), 'agg_rmse': rmse(p_official, yt), 'opp_rmse': rmse(p_opp, yt),
                     'gap_agg_minus_opp': rmse(p_official, yt) - rmse(p_opp, yt)},
        'label_gt_censor': {'n': int((~mask_t).sum()), 'agg_rmse': rmse(p_official[~mask_t], yt[~mask_t]),
                            'opp_rmse': rmse(p_opp[~mask_t], yt[~mask_t]),
                            'gap_agg_minus_opp': rmse(p_official[~mask_t], yt[~mask_t]) - rmse(p_opp[~mask_t], yt[~mask_t])},
        'label_eq_censor': {'n': int(mask_t.sum()), 'agg_rmse': rmse(p_official[mask_t], yt[mask_t]),
                            'opp_rmse': rmse(p_opp[mask_t], yt[mask_t]),
                            'gap_agg_minus_opp': rmse(p_official[mask_t], yt[mask_t]) - rmse(p_opp[mask_t], yt[mask_t])},
    }

    # (b) 用同样的 caliber（simplex/nnls）在 val 的「剔 5.0」子集上重拟合 θ，再在 test 各子集评估
    caliber = 'simplex' if 'simplex' in (row.get('caliber') or '') else 'nnls_free'
    fitter = fit_simplex if caliber == 'simplex' else fit_nnls
    th_refit = fitter(Pv[~mask_v], yv[~mask_v])
    p_refit = apply_th(th_refit, Pt)
    refit = {
        'caliber': caliber,
        'theta_refit_on_val_label_gt_censor': [float(x) for x in th_refit],
        'n_val_used': int((~mask_v).sum()),
        'all_rows': {'agg_rmse': rmse(p_refit, yt)},
        'label_gt_censor': {'agg_rmse': rmse(p_refit[~mask_t], yt[~mask_t])},
        'label_eq_censor': {'agg_rmse': rmse(p_refit[mask_t], yt[mask_t])},
    }

    # (c) 常数预测器
    intercept = float(th_official[0])
    consts = {
        'intercept_1p398_alone': {
            'value': intercept,
            'test_rmse_all': rmse(np.full(len(yt), intercept), yt),
            'test_rmse_label_gt_censor': rmse(np.full(int((~mask_t).sum()), intercept), yt[~mask_t]),
            'test_rmse_label_eq_censor': rmse(np.full(int(mask_t.sum()), intercept), yt[mask_t]),
            'val_rmse': rmse(np.full(len(yv), intercept), yv),
        },
        'column_mean_train': {
            'value': float(sub_t['pK'].mean() * 0 + dp[dp[ckey] == 'train'].pK.mean()),
            'definition': '训练划分 pK 列均值（无标签常数预测器）',
            'test_rmse_all': None, 'test_rmse_label_gt_censor': None, 'test_rmse_label_eq_censor': None,
            'val_rmse': None,
        },
        'column_mean_test_oracle': {
            'value': float(yt.mean()),
            'definition': '测试划分 pK 列均值（用到 test 标签，oracle 参照）',
            'test_rmse_all': None, 'test_rmse_label_gt_censor': None, 'test_rmse_label_eq_censor': None,
            'val_rmse': None,
        },
        'column_mean_val': {
            'value': float(yv.mean()),
            'definition': '验证划分 pK 列均值',
            'test_rmse_all': None, 'test_rmse_label_gt_censor': None, 'test_rmse_label_eq_censor': None,
            'val_rmse': None,
        },
    }
    for key in ['column_mean_train', 'column_mean_test_oracle', 'column_mean_val']:
        c = consts[key]['value']
        consts[key]['test_rmse_all'] = rmse(np.full(len(yt), c), yt)
        consts[key]['test_rmse_label_gt_censor'] = rmse(np.full(int((~mask_t).sum()), c), yt[~mask_t])
        consts[key]['test_rmse_label_eq_censor'] = rmse(np.full(int(mask_t.sum()), c), yt[mask_t])
        consts[key]['val_rmse'] = rmse(np.full(len(yv), c), yv)

    # (d) 化合物聚类 bootstrap（整簇重采样，等量权重）
    sub = sub_t
    smiles_col = np.asarray(sub['smiles'].values, dtype=object)
    uniq = sorted(set(smiles_col.tolist()))
    counts = {u: int((smiles_col == u).sum()) for u in uniq}
    e_agg = (p_official - yt) ** 2
    e_opp = (p_opp - yt) ** 2
    per_comp = {}
    for u in uniq:
        m = smiles_col == u
        per_comp[u] = {'n': int(m.sum()), 'mse_agg': float(e_agg[m].mean()), 'mse_opp': float(e_opp[m].mean()),
                       'dMSE_agg_minus_opp': float(e_agg[m].mean() - e_opp[m].mean())}
    K = len(uniq)
    idx_list = [(smiles_col == u).nonzero()[0] for u in uniq]
    sizes = np.array([len(i) for i in idx_list], float)
    s_agg = np.array([e_agg[i].sum() for i in idx_list])
    s_opp = np.array([e_opp[i].sum() for i in idx_list])
    n_tot = sizes.sum()
    if K <= 8:
        from itertools import product
        dmse = np.array([(s_agg[list(c)].sum() - s_opp[list(c)].sum()) / n_tot
                         for c in product(range(K), repeat=K)])
        scheme = ('整簇重采样 %d^%d = %d 次全枚举（每化合物所有靶点行一起重采样；'
                  '每簇采 K 次，权重按行数等量）' % (K, K, K ** K))
    else:
        # K 太大（davis_cold 有 68 化合物 → 68^68 不可枚举）：多项分布整簇重采样
        rng = np.random.RandomState(2026)
        NB = 200000
        cnt = rng.multinomial(K, np.full(K, 1.0 / K), size=NB)  # (NB, K)
        dmse = (cnt @ s_agg - cnt @ s_opp) / n_tot
        scheme = ('整簇重采样：多项分布 K=%d 簇抽取 K 个（等量概率），%d 次 Monte Carlo '
                  '（%d^%d 不可枚举）' % (K, NB, K, K))
    boot = {
        'unit': 'compound (SMILES) cluster, %d unique compounds in test' % K,
        'scheme': scheme,
        'n_resamples': int(len(dmse)),
        'per_compound_row_counts': counts,
        'per_compound_mse': per_comp,
        'sign_convention': 'dMSE = MSE_agg - MSE_opponent（>0 表示 Kairos 更差）',
        'mean_dMSE': float(dmse.mean()),
        'frac_dMSE_le_0': float((dmse <= 0).mean()),
        'pct2.5': float(np.percentile(dmse, 2.5)),
        'pct97.5': float(np.percentile(dmse, 97.5)),
        'observed_dMSE_point': float(mse(p_official, yt) - mse(p_opp, yt)),
    }

    recon[cell] = {'n_test': int(len(yt)), 'opponent': opp_name, 'members': members,
                   'fixed_theta': fixed, 'refit_theta_excl_censor': refit,
                   'constant_predictors': consts, 'compound_cluster_bootstrap': boot}
    prog('%s done' % cell)

censor['rmse_reconstruction'] = recon

# ------------------------------------------- 2b. davis_official（官方 setting1）删失重建
# 该格 matrix 的 theta 为 null，口径是 cv5（旧成员无 val 预测），
# 与 results/absorb/p52_ceiling_davis.json 的 cv5_aggregate 同口径。
z = np.load(os.path.join(MID, 'sota_duel', 'ext', 'davis_member_preds_aligned.npz'), allow_pickle=True)
old = ['esmmax', 'lightdta', 'deepdta', 'dtaom', 'e3mmdta', 'mutualdta']
off_dir = os.path.join(MID, 'results', 'p2', 'davis_official')
new = sorted({f.split('pred_')[1].split('_test')[0] for f in os.listdir(off_dir)
              if f.startswith('pred_') and f.endswith('_test.parquet')})
cols, names, y_off = [], [], np.asarray(z['y'], float)
for m in old:
    cols.append(np.asarray(z[m], float)); names.append(m)
for m in new:
    t = pd.read_parquet(os.path.join(off_dir, 'pred_%s_test.parquet' % m))
    assert len(t) == len(y_off) and np.allclose(t['pK'].values, y_off), 'davis_official 行序错位: ' + m
    cols.append(t['pred'].values.astype(float)); names.append(m)
Poff = np.column_stack(cols)


def cv5(P, y, seed=42):
    rng = np.random.RandomState(seed)
    fold = rng.permutation(len(y)) % 5
    oof = np.zeros(len(y))
    for f in range(5):
        tr = fold != f
        th = fit_nnls(P[tr].astype(np.float64), y[tr].astype(np.float64))
        oof[~tr] = apply_th(th, P[~tr])
    return oof


p_cv5 = cv5(Poff, y_off)
mt = (y_off == CENSOR)
opp_name_off = ROWW['Davis|随机']['opponent']['name']
p_opp_off = Poff[:, names.index(opp_name_off)]
cens_off = {
    'cell': 'davis_official (Davis 官方 setting1 单划分)',
    'n_test': int(len(y_off)), 'members': names,
    'caliber': 'cv5（test 内 5 折交叉拟合 nnls；与 results/absorb/p52_ceiling_davis.json 同口径，'
               '因旧成员无 val 预测，matrix 该格 theta=null）',
    'cv5_aggregate_rmse_rebuilt': rmse(p_cv5, y_off),
    'frozen_cv5_aggregate_rmse': 0.47964719221232316,
    'reproduces_frozen': bool(abs(rmse(p_cv5, y_off) - 0.47964719221232316) < 1e-6),
    'opponent': opp_name_off,
    'subsets': {
        'all_rows': {'n': int(len(y_off)), 'agg_rmse': rmse(p_cv5, y_off), 'opp_rmse': rmse(p_opp_off, y_off),
                     'gap': rmse(p_cv5, y_off) - rmse(p_opp_off, y_off)},
        'label_gt_censor': {'n': int((~mt).sum()), 'agg_rmse': rmse(p_cv5[~mt], y_off[~mt]),
                            'opp_rmse': rmse(p_opp_off[~mt], y_off[~mt]),
                            'gap': rmse(p_cv5[~mt], y_off[~mt]) - rmse(p_opp_off[~mt], y_off[~mt])},
        'label_eq_censor': {'n': int(mt.sum()), 'agg_rmse': rmse(p_cv5[mt], y_off[mt]),
                            'opp_rmse': rmse(p_opp_off[mt], y_off[mt]),
                            'gap': rmse(p_cv5[mt], y_off[mt]) - rmse(p_opp_off[mt], y_off[mt])},
    },
    'injectivity_caveat': '该格成员含 dtaom/e3mmdta/mutualdta 等公开声称复现方法；此处只做删失子集分解，不改变论文的复现结论。',
}
censor['rmse_reconstruction']['davis_official'] = cens_off
prog('davis_official done')

json.dump(censor, open(os.path.join(OUT, 'davis_censoring.json'), 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print(json.dumps(censor, ensure_ascii=False, indent=1))
print('\nWROTE davis_censoring.json')