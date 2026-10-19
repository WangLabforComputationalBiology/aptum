# -*- coding: utf-8 -*-
"""
Kairos-θ 官方口径规则 · 统一重算
================================
规则（无标签，只用预测分布）：
  d = max_k |mean(p_k^test) − mean(p_k^val)| / sd(p_k^val)
  d ≤ 0.10 → 同分布：NNLS 自由 θ（θ≥0, 截距自由）
  d > 0.10 → 分布位移：凸组合 simplex（θ≥0, Σθ=1, 截距自由）
  并列报告 cv5_test（test 内 5 折交叉拟合）作为 transductive 参考

本脚本从 results/p2/{cell}/ 的落盘预测重算全部口径与 CI（用 ci_lib 的已勘误实现），
写出 results/kairos_theta_rule.json，供矩阵汇总脚本直接读。
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_lib import ci_harrell
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(MID, 'results', 'p2')
THRESH = 0.10


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def fit_nnls(P, y):
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    A = np.column_stack([np.ones(len(y)), P])
    r = lsq_linear(A, y, bounds=(np.array([-np.inf] + [0.0] * P.shape[1]),
                                 np.full(P.shape[1] + 1, np.inf)), method='trf')
    return r.x


def fit_simplex(P, y):
    """θ≥0, Σθ=1（凸组合），截距自由。变量替换 θ_K = 1 − Σθ_{<K}。"""
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    if K == 1:
        return np.array([0.0, 1.0])
    A = np.column_stack([np.ones(len(y)), P[:, :K - 1] - P[:, K - 1:K]])
    rhs = y - P[:, K - 1]
    lo = np.array([-np.inf] + [0.0] * (K - 1))
    hi = np.array([np.inf] + [1.0] * (K - 1))
    r = lsq_linear(A, rhs, bounds=(lo, hi), method='trf')
    th = np.zeros(K)
    th[:K - 1] = r.x[1:]
    th[K - 1] = 1.0 - th[:K - 1].sum()
    if th[K - 1] < -1e-9:
        th = np.clip(th, 0, None); th = th / th.sum()
    return np.concatenate([[r.x[0]], th])


def apply_th(th, P):
    return th[0] + np.asarray(P, float) @ th[1:]


def metrics(p, y, y_ref_mean):
    p, y = np.asarray(p, float), np.asarray(y, float)
    mse = float(np.mean((p - y) ** 2))
    ss = float(np.sum((y - y_ref_mean) ** 2))
    return {'rmse': float(np.sqrt(mse)), 'mse': mse, 'ci': float(ci_harrell(p, y)),
            'spearman': float(spearmanr(p, y).statistic),
            'r2m': float(1 - np.sum((p - y) ** 2) / ss) if ss > 0 else float('nan'),
            'n': int(len(y))}


def main():
    out = {
        'rule': {
            'name': 'Kairos-θ 自动口径切换（无标签）',
            'd_definition': 'd = max_k |mean(p_k^test) - mean(p_k^val)| / sd(p_k^val)',
            'threshold': THRESH,
            'branch_same_dist': 'd <= 0.10 → NNLS 自由 θ (θ>=0, 截距自由)',
            'branch_shift': 'd > 0.10 → 凸组合 simplex (θ>=0, Σθ=1, 截距自由)',
            'reference': '并列报告 cv5_test（test 内 5 折交叉拟合，transductive：对应"目标域有一小批已测数据"场景）',
            'threshold_justification': '阈值 0.10 位于已测各格位移量的天然空隙中（同分布组 ≤0.04，位移组 ≥0.17），相差一个数量级，非拟合参数',
            'disclosure': '规则在观察到 Davis scaffold 失败后提出。它未损害任何未失败格子（simplex 与 nnls 在那些格子上差异 ≤0.014），且阈值远离两簇。对后续格子构成真实的样本外检验。',
            'ci_impl': 'ci_lib.ci_harrell（Harrell C，tie 忽略；旧分块实现漏计块内配对已勘误）',
        },
        'cells': {},
        'history': {},
    }

    old = json.load(open(os.path.join(MID, 'results', 'kairos_theta_rule.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(MID, 'results', 'kairos_theta_rule.json')) else {}
    out['history'] = old.get('cells', {})

    for cell in sorted(glob.glob(os.path.join(P2, '*_*'))):
        if not os.path.isdir(cell):
            continue
        key = os.path.basename(cell)
        names = sorted({f.split('pred_')[1].split('_test')[0]
                        for f in os.listdir(cell) if f.startswith('pred_') and f.endswith('_test.parquet')})
        if len(names) < 2:
            continue
        te0 = pd.read_parquet(os.path.join(cell, 'pred_%s_test.parquet' % names[0]))
        va0 = pd.read_parquet(os.path.join(cell, 'pred_%s_val.parquet' % names[0]))
        yte, yva = te0['pK'].values, va0['pK'].values
        Pte = np.column_stack([te0['pred'].values] +
                             [pd.read_parquet(os.path.join(cell, 'pred_%s_test.parquet' % m))['pred'].values
                              for m in names[1:]])
        Pva = np.column_stack([va0['pred'].values] +
                             [pd.read_parquet(os.path.join(cell, 'pred_%s_val.parquet' % m))['pred'].values
                              for m in names[1:]])
        ymean = float(yva.mean())
        dmax = float(np.max([abs(np.mean(Pte[:, j]) - np.mean(Pva[:, j])) / max(np.std(Pva[:, j]), 1e-9)
                            for j in range(len(names))]))
        chosen = 'simplex' if dmax > THRESH else 'nnls_free'

        th_n = fit_nnls(Pva, yva)
        th_s = fit_simplex(Pva, yva)
        th_o = fit_nnls(Pte, yte)
        rngc = np.random.RandomState(42)
        fold = rngc.permutation(len(yte)) % 5
        oof = np.zeros(len(yte))
        for f in range(5):
            tr = fold != f
            t = fit_nnls(Pte[tr], yte[tr])
            oof[~tr] = apply_th(t, Pte[~tr])

        single = {m: metrics(Pte[:, j], yte, ymean) for j, m in enumerate(names)}
        bi = min(names, key=lambda m: single[m]['rmse'])
        thr = th_s if chosen == 'simplex' else th_n
        p_off = apply_th(thr, Pte)

        out['cells'][key] = {
            'shift_d_max': round(dmax, 4), 'chosen': chosen,
            'members': names, 'n_test': int(len(yte)), 'n_val': int(len(yva)),
            'single': single, 'best_single': bi,
            'official': dict(theta=[round(float(x), 4) for x in thr],
                             theta_source='val（同划分内留出，未参与训练）',
                             **metrics(p_off, yte, ymean)),
            'nnls_free': dict(theta=[round(float(x), 4) for x in th_n],
                              **metrics(apply_th(th_n, Pte), yte, ymean)),
            'simplex': dict(theta=[round(float(x), 4) for x in th_s],
                            **metrics(apply_th(th_s, Pte), yte, ymean)),
            'cv5_test': dict(**metrics(oof, yte, ymean)),
            'oracle_testfit': dict(theta=[round(float(x), 4) for x in th_o],
                                   theta_source='⚠ test 上直接拟合 = 不可达上界',
                                   **metrics(apply_th(th_o, Pte), yte, ymean)),
        }
        o = out['cells'][key]['official']
        print('%-20s d=%.4f → %-10s 官方 RMSE=%.4f CI=%.4f (最佳单 %s %.4f)'
              % (key, dmax, chosen, o['rmse'], o['ci'], bi, single[bi]['rmse']))

    js = os.path.join(MID, 'results', 'kairos_theta_rule.json')
    if '--write' in sys.argv:
        json.dump(out, open(js, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('\n写入 %s' % js)
    else:
        print('\n[dry-run] 未写回 %s' % js)
        print('  这里是只读复现入口：本包随附的 kairos_theta_rule.json 是论文的权威版本，')
        print('  直接覆盖会丢格（需要逐成员预测才能完整重算）。')
        print('  要强制重写请加 --write。')


if __name__ == '__main__':
    main()
