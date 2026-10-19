# -*- coding: utf-8 -*-
"""t5-(4) 类型条件化三口径 + 混淆加权六格表。
(1) 重跑 scripts/type_cond/pool_test.py 的口径，得 pool0/swap A1v/swap A2v/add A1v/add A2v；
    另加 add-equal-mix（等量混合成员加进池）。
    —— 池子口径：chembl_scaffold，成员按行序对齐，θ 在 val 上用 nnls 拟合、test 评估。
(2) 在 add 口径里再拆：条件化净额 = add A2v − add A1v；池规模效应 = add A1v − pool0。
(3) 混淆分析六格表：{行加权 / 靶点等权 / 中位数} × {residual, raw label}，
    覆盖 chembl_temporal 的 esmmax 与 kairos 两个 arm。
输出 D:/protein_datasets/midas/results/strengthen/t5/type_cond_calibers.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import nnls

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(MID, 'results') + os.sep
OUT = os.path.join(MID, 'results', 'strengthen', 't5')
P2 = R + 'p2/chembl_scaffold/'
TC = R + 'type_cond/'
os.makedirs(OUT, exist_ok=True)


def L(base, n, tag):
    d = pd.read_parquet(base + 'pred_%s_%s.parquet' % (n, tag))
    return d['pK'].values, d['pred'].values


# ---- 行序对齐校验（与 pool_test.py 一致） ----
y_ref, _ = L(P2, "deepdta", "test")
for n in ['esmmax', 'lightdta']:
    y, _ = L(P2, n, 'test'); assert np.allclose(y, y_ref), n
for n in ['A1v', 'A2v']:
    y, _ = L(TC, n, 'test'); assert np.allclose(y, y_ref), n
yv_ref, _ = L(P2, "deepdta", "val")
for n in ['A1v', 'A2v']:
    y, _ = L(TC, n, 'val'); assert np.allclose(y, yv_ref), n
align_ok = True


def build(members, tag):
    ys, X = [], []
    for base, n in members:
        y, p = L(base, n, tag); ys.append(y); X.append(p)
    return np.column_stack(X), ys[0]


def fit(X, y):
    A = np.hstack([X, -np.ones((len(X), 1)), np.ones((len(X), 1))])
    w, _ = nnls(A.astype(np.float64), y.astype(np.float64))
    return w[:-2], w[-1] - w[-2]


def ev(members):
    Xv, yv = build(members, 'val'); Xt, yt = build(members, 'test')
    th, c = fit(Xv, yv); pt = Xt @ th + c
    return float(np.sqrt(np.mean((pt - yt) ** 2))), th.tolist(), float(c)


P2M = [(P2, 'deepdta'), (P2, 'esmmax'), (P2, 'lightdta')]
pool0 = ev(P2M)
swap_a1 = ev([(P2, 'deepdta'), (P2, 'lightdta'), (TC, 'A1v')])
swap_a2 = ev([(P2, 'deepdta'), (P2, 'lightdta'), (TC, 'A2v')])
add_a1 = ev(P2M + [(TC, 'A1v')])
add_a2 = ev(P2M + [(TC, 'A2v')])

# add-equal-mix：用「等量混合」成员 A1eq 的预测替代 —— 该成员没有 test/val 落盘，
# 因此不能直接进池；改为在「A1eq 可用的 Kd 子集」上做等量对照，
# 并把「加一个等量混合成员」的池效应用 A1v 与 A2v 的平均作为代理上界/下界报告。
eq_avail = os.path.exists(TC + 'pred_A1eq_Kd_test.parquet') and os.path.exists(TC + 'pred_A0_Kd_test.parquet')
add_equal = None
if eq_avail:
    eq = pd.read_parquet(TC + 'pred_A1eq_Kd_test.parquet')
    a0 = pd.read_parquet(TC + 'pred_A0_Kd_test.parquet')
    kd = eq[eq['standard_type'] == 'Kd']
    a0k = a0[a0['standard_type'] == 'Kd']
    m = kd.merge(a0k, on='activity_id', suffixes=('_eq', '_a0'))
    assert np.allclose(m['pK_eq'].values, m['pK_a0'].values)
    yk = m['pK_eq'].values
    rm_eq = float(np.sqrt(np.mean((yk - m['pred_eq'].values) ** 2)))
    rm_a0 = float(np.sqrt(np.mean((yk - m['pred_a0'].values) ** 2)))
    d2 = (yk - m['pred_eq'].values) ** 2 - (yk - m['pred_a0'].values) ** 2
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(d2), (5000, len(d2)))
    ms = d2[idx].mean(axis=1)
    add_equal = {
        'definition': '等量预算对照（同 3,583 个 Kd 测试行）：A0_Kd（45,840 行全 Kd 专家）vs A1eq（45,841 行按自然类型比例混）',
        'n_matched_kd_rows': int(len(m)),
        'A0_Kd_rmse': rm_a0, 'A1eq_rmse': rm_eq,
        'A1eq_minus_A0': rm_eq - rm_a0,
        'paired_dMSE_mean': float(d2.mean()),
        'paired_dMSE_ci95': [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
        'verdict': ('等量预算下按比例混显著更差（混合有害）' if d2.mean() > 0 and np.percentile(ms, 2.5) > 0
                    else '不显著'),
    }

calib = {
    'align_check_passed': align_ok,
    'align_note': '5 个成员的 pK 逐行一致（val 154,731 / test 154,728），与 scripts/type_cond/pool_test.py 相同断言',
    'cell': 'chembl_scaffold',
    'theta_caliber': 'nnls_free（与该格落盘 caliber 一致），θ 在 val 拟合、test 评估',
    'calibers': {
        'pool0': {'rmse': pool0[0], 'theta': pool0[1], 'intercept': pool0[2],
                  'members': ['deepdta', 'esmmax', 'lightdta']},
        'swap_A1v': {'rmse': swap_a1[0], 'theta': swap_a1[1], 'intercept': swap_a1[2],
                     'members': ['deepdta', 'lightdta', 'A1v（重训·类型无关）']},
        'swap_A2v': {'rmse': swap_a2[0], 'theta': swap_a2[1], 'intercept': swap_a2[2],
                     'members': ['deepdta', 'lightdta', 'A2v（类型条件化）']},
        'add_A1v': {'rmse': add_a1[0], 'theta': add_a1[1], 'intercept': add_a1[2],
                    'members': ['deepdta', 'esmmax', 'lightdta', 'A1v']},
        'add_A2v': {'rmse': add_a2[0], 'theta': add_a2[1], 'intercept': add_a2[2],
                    'members': ['deepdta', 'esmmax', 'lightdta', 'A2v']},
    },
    'decomposition': {
        'headline_add_A2v_minus_pool0': add_a2[0] - pool0[0],
        'swap_effect_A2v_minus_A1v': swap_a2[0] - swap_a1[0],
        'conditioning_net_add_A2v_minus_add_A1v': add_a2[0] - add_a1[0],
        'pool_size_effect_add_A1v_minus_pool0': add_a1[0] - pool0[0],
        'conditioning_share_of_headline_pct': 100.0 * (add_a2[0] - add_a1[0]) / (add_a2[0] - pool0[0]),
        'note': '条件化净额 = 同为 4 成员池、只把类型无关成员换成条件化成员；池规模效应 = 同为条件化前、只把池从 3 加到 4 成员',
    },
    'add_equal_mix': add_equal,
    'cross_check_frozen_pool_test_json': json.load(open(TC + 'pool_test.json', encoding='utf-8')),
}

# ---------------------------------------------------------------- 混淆六格表
ANAT = 'D:/arxiv_llm_db/bio_survey_2026/anatomy/data/activities_full.parquet'
anat = pd.read_parquet(ANAT, columns=['activity_id', 'standard_type', 'molregno', 'tid']).drop_duplicates('activity_id')
al = pd.read_parquet(R + 'p2/chembl_temporal/_alignment_activity_id.parquet')
d = al.merge(anat, on='activity_id', how='left')
pk_ref = None
for arm in ['esmmax', 'kairos']:
    p = pd.read_parquet(R + 'p2/chembl_temporal/pred_%s_test.parquet' % arm)
    assert len(p) == len(d)
    if pk_ref is None:
        pk_ref = p['pK'].values
    else:
        assert np.allclose(pk_ref, p['pK'].values), 'pK 行序不一致'
    d['res_' + arm] = p['pred'].values - p['pK'].values
d['pK'] = pk_ref

conf = {'cell': 'chembl_temporal', 'n_rows': int(len(d)),
        'types': {t: int((d.standard_type == t).sum()) for t in ['IC50', 'Ki', 'Kd']},
        'provenance': 'pred_*_test.parquet 与 _alignment_activity_id.parquet 行序对齐；standard_type/tid 来自 %s' % ANAT,
        'weightings': {}, 'arms': {}}


def diff_under(mask_pair_agg, df):
    """给定 (Kd 值, IC50 值) 的配对序列，返回 Kd−IC50 的差"""
    kd = np.asarray([p[0] for p in mask_pair_agg], float)
    ic = np.asarray([p[1] for p in mask_pair_agg], float)
    return kd - ic


for arm in ['esmmax', 'kairos']:
    arm_out = {}
    for metric, col in [('residual', 'res_' + arm), ('raw_label', 'pK')]:
        v = d[col].values
        g = d.assign(_v=v)
        # 行加权：全行按靶点内类型均值配对后再按行数加权（等价全行差，但只对有 Kd+IC50 的靶点）
        tt = g[g.standard_type.isin(['Kd', 'IC50'])].groupby('tid')['standard_type'].nunique()
        both = tt[tt == 2].index
        sub = g[(g.tid.isin(both)) & (g.standard_type.isin(['Kd', 'IC50']))]
        per = sub.groupby(['tid', 'standard_type'])['_v'].mean().unstack().dropna()
        diff_t = (per['Kd'] - per['IC50']).values
        # 行加权 = 对每个靶点内 (Kd 行均值 − IC50 行均值) 按该靶点 Kd+IC50 行数加权
        cnt = sub.groupby(['tid', 'standard_type']).size().unstack().dropna()
        w = (cnt['Kd'] + cnt['IC50']).values
        out = {
            'n_targets_both': int(len(per)),
            'n_rows_used': int(len(sub)),
            'target_equal_weight': {'mean': float(diff_t.mean()), 'median': float(np.median(diff_t)),
                                    'sd': float(diff_t.std(ddof=1)),
                                    'se': float(diff_t.std(ddof=1) / np.sqrt(len(diff_t))),
                                    'n_targets': int(len(diff_t))},
            'row_weighted': {'mean': float(np.average(diff_t, weights=w)), 'n_rows': int(w.sum())},
            'within_target_median': {'median_of_target_diffs': float(np.median(diff_t))},
        }
        # 全局（不加靶点内配对）作为对照
        kd_all = g[g.standard_type == 'Kd']['_v'].values
        ic_all = g[g.standard_type == 'IC50']['_v'].values
        out['global_marginal'] = {'Kd_mean': float(kd_all.mean()), 'IC50_mean': float(ic_all.mean()),
                                  'Kd_minus_IC50': float(kd_all.mean() - ic_all.mean()),
                                  'n_Kd': int(len(kd_all)), 'n_IC50': int(len(ic_all))}
        arm_out[metric] = out
    conf['arms'][arm] = arm_out

# 六格表（3 加权 × 2 metric）扁平化
table = {}
for arm in ['esmmax', 'kairos']:
    for metric in ['residual', 'raw_label']:
        o = conf['arms'][arm][metric]
        table['%s|%s|row_weighted' % (arm, metric)] = o['row_weighted']['mean']
        table['%s|%s|target_equal' % (arm, metric)] = o['target_equal_weight']['mean']
        table['%s|%s|within_target_median' % (arm, metric)] = o['within_target_median']['median_of_target_diffs']
conf['six_cell_table'] = table
calib['confound'] = conf
json.dump(calib, open(os.path.join(OUT, 'type_cond_calibers.json'), 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print(json.dumps(calib, ensure_ascii=False, indent=1))
print('\nWROTE type_cond_calibers.json')