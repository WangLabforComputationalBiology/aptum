# -*- coding: utf-8 -*-
"""
t2 · 求解器与单纯形约束对账  (只读输入，新产物写 results/strengthen/t2/)
=====================================================================
15 格（= results/matrix_4x5.json 中 theta 非空的全部行）× 5 条单纯形求解路径：
  A clip_trf   scripts/kairos_rule.py::fit_simplex 现行实现（变量替换 + 第 54-55 行 clip 归一）
  B qp_slsqp   真约束 QP：SLSQP, constraints=[Σθ=1], bounds=[(-inf,inf)]+[(0,1)]*K, 10 个 Dirichlet 随机起点
  C trf_bound  真约束 trf（变量替换 + θ_i ≤ 1 上界，无 clip）
  D lsq_bvls   lsq_linear(method='bvls')
  E dirichlet  2e5 次 Dirichlet 随机搜索（闭式最优截距，全局网格搜索）

闭式恒等式（惩罚 0、截距自由时的精确最优截距 b* = ȳ − P̄θ）：
  SSE(θ) = ||Y − Pc θ||²  ,  Y = y − ȳ ,  Pc = P − P̄(行方向中心化)
         = YY − 2 θᵀg + θᵀ A θ  ,  A = PcᵀPc , g = PcᵀY
因此在单纯形上最小化 SSE 等价于最小化凸二次型 θᵀAθ − 2θᵀg。本脚本用该恒等式
（a）给 Dirichlet 搜索提供 O(K²)/draw 的精确目标函数，（b）作为对各求解器设解的最优性判据。
"""
import os, sys, json, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_nnls, fit_simplex, apply_th, THRESH
from ci_lib import ci_harrell
from scipy.optimize import minimize, lsq_linear

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(MID, 'results', 'p2')
OUT = os.path.join(MID, 'results', 'strengthen', 't2')
os.makedirs(OUT, exist_ok=True)

N_DIR = 200_000
N_START = 10

# matrix_4x5.json 的 (dataset, protocol) → p2 目录名
CELLMAP = [
    ('ChEMBL 战场', '随机', 'chembl_random'),
    ('ChEMBL 战场', 'scaffold', 'chembl_scaffold'),
    ('ChEMBL 战场', '冷启动', 'chembl_cold'),
    ('BindingDB', '时间', 'bindingdb_temporal'),
    ('BindingDB', '随机', 'bindingdb_random'),
    ('BindingDB', 'scaffold', 'bindingdb_scaffold'),
    ('BindingDB', '冷启动', 'bindingdb_cold'),
    ('Davis', 'scaffold', 'davis_scaffold'),
    ('Davis', '冷启动', 'davis_cold'),
    ('KIBA', '随机', 'kiba_official'),
    ('KIBA', 'scaffold', 'kiba_scaffold'),
    ('KIBA', '冷启动', 'kiba_cold'),
    ('激酶基准', '随机', 'kinase_random'),
    ('激酶基准', 'scaffold', 'kinase_scaffold'),
    ('激酶基准', '冷启动', 'kinase_target'),
]


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def load_cell(cell):
    d = os.path.join(P2, cell)
    names = sorted({f.split('pred_')[1].split('_test')[0]
                    for f in os.listdir(d) if f.startswith('pred_') and f.endswith('_test.parquet')})
    te0 = pd.read_parquet(os.path.join(d, 'pred_%s_test.parquet' % names[0]))
    va0 = pd.read_parquet(os.path.join(d, 'pred_%s_val.parquet' % names[0]))
    yte, yva = te0['pK'].values.astype(np.float64), va0['pK'].values.astype(np.float64)
    Pte = np.column_stack([pd.read_parquet(os.path.join(d, 'pred_%s_test.parquet' % m))['pred'].values
                           for m in names]).astype(np.float64)
    Pva = np.column_stack([pd.read_parquet(os.path.join(d, 'pred_%s_val.parquet' % m))['pred'].values
                           for m in names]).astype(np.float64)
    # 行对齐自检：所有成员的 parquet 里 pK 必须一致
    align_ok = True
    for m in names[1:]:
        if not np.array_equal(pd.read_parquet(os.path.join(d, 'pred_%s_test.parquet' % m))['pK'].values, yte):
            align_ok = False
        if not np.array_equal(pd.read_parquet(os.path.join(d, 'pred_%s_val.parquet' % m))['pK'].values, yva):
            align_ok = False
    return names, Pva, yva, Pte, yte, align_ok


def quad_cache(P, y):
    """A = PcᵀPc, g = PcᵀY, YY, ȳ, P̄ —— 供闭式 SSE 使用。"""
    ybar = float(y.mean()); Py = P.mean(0)
    Y = y - ybar; Pc = P - Py
    return {'A': Pc.T @ Pc, 'g': Pc.T @ Y, 'YY': float(Y @ Y), 'ybar': ybar, 'Pbar': Py, 'N': len(y)}


def sse_q(cache, TH):
    """TH: m×K → m 个 SSE（未除 N）。"""
    return cache['YY'] - 2.0 * (TH @ cache['g']) + np.einsum('ij,ij->i', TH, TH @ cache['A'])


# ---------------------------------------------------------------- A: 现行 clip 路径
def fit_simplex_clip(P, y):
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    A = np.column_stack([np.ones(len(y)), P[:, :K - 1] - P[:, K - 1:K]])
    rhs = y - P[:, K - 1]
    lo = np.array([-np.inf] + [0.0] * (K - 1))
    hi = np.array([np.inf] + [1.0] * (K - 1))
    r = lsq_linear(A, rhs, bounds=(lo, hi), method='trf')
    th = np.zeros(K); th[:K - 1] = r.x[1:]; th[K - 1] = 1.0 - th[:K - 1].sum()
    pre = float(th[:K - 1].sum())
    clip = bool(th[K - 1] < -1e-9)
    d = {'sum_pre_clip': pre, 'clip_triggered': clip,
         'bound_hit_theta_le_1': int(np.sum(np.abs(r.x[1:] - 1.0) < 1e-9)),
         'sum_pre_clip_minus1': pre - 1.0}
    if clip:
        th = np.clip(th, 0, None); th = th / th.sum()
    return np.concatenate([[r.x[0]], th]), d


# ---------------------------------------------------------------- B: 真约束 QP (SLSQP)
def fit_simplex_slsqp(P, y, n_start=N_START, seed=0):
    """θ≥0, Σθ=1, 截距自由。

    截距 b 在给定 θ 下的精确最优是 b*(θ) = ȳ − P̄θ（因 SSE 对 b 是二次的、无约束最小在
    残差均值为零处）。把它代进目标，则优化变量只剩 θ（K 维），SLSQP 的变量与约束都是 K 维；
    最后按 b*(θ) 回填截距——这正是"截距自由"下的精确解，避免把 b 当无关变量留给求解器。
    """
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    c = quad_cache(P, y)
    rng = np.random.RandomState(seed)

    def obj(th):
        return float(c['YY'] - 2.0 * th @ c['g'] + th @ c['A'] @ th)

    cons = [{'type': 'eq', 'fun': lambda th: float(np.sum(th) - 1.0)}]
    bounds = [(0.0, 1.0)] * K
    starts = [np.full(K, 1.0 / K)]
    th_n = fit_nnls(P, y)
    t0 = np.clip(th_n[1:], 0, None)
    if t0.sum() > 0:
        starts.append(t0 / t0.sum())
    while len(starts) < n_start:
        starts.append(rng.dirichlet(np.ones(K)))
    best, bestf, nfail = None, np.inf, 0
    for s in starts[:n_start]:
        try:
            r = minimize(obj, s, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 1000, 'ftol': 1e-14})
        except Exception:
            nfail += 1; continue
        if r.fun < bestf:
            bestf, best = float(r.fun), r.x.copy()
    best = np.clip(best, 0, None); best = best / best.sum()
    b = c['ybar'] - float(c['Pbar'] @ best)
    return np.concatenate([[b], best]), {'slsqp_starts': n_start, 'slsqp_fail': nfail}


# ------------------------------------------------- B2: 精确 active-set QP（独立校验，非论文口径，只用于验证 SLSQP 是否全局最优）
def fit_simplex_exact(P, y, max_faces=1 << 16):
    """单纯形上凸 QP 的精确全局最优：枚举全部支撑面，在每面上解等式约束 KKT，
    只接受内部可行解，取目标最小者。KKT 对凸 QP + 仿射约束是充要条件 → 全局最优。"""
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    c = quad_cache(P, y)
    A, g = c['A'], c['g']
    nfaces = 0
    best, bestf = None, np.inf
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            nfaces += 1
            if nfaces > max_faces:
                return best, {'faces': nfaces, 'truncated': True, 'n_support': int((best > 0).sum())}
            idx = list(S); k = len(idx)
            As = A[np.ix_(idx, idx)]; gs = g[idx]; ones = np.ones(k)
            M = np.zeros((k + 1, k + 1))
            M[:k, :k] = 2.0 * As
            M[:k, k] = ones; M[k, :k] = ones
            try:
                sol = np.linalg.solve(M, np.concatenate([2.0 * gs, [1.0]]))
            except np.linalg.LinAlgError:
                continue
            th = sol[:k]
            if np.any(th < -1e-9):
                continue
            t = np.zeros(K); t[idx] = np.clip(th, 0, None)
            if t.sum() <= 0:
                continue
            t = t / t.sum()
            f = float(c['YY'] - 2.0 * t @ g + t @ A @ t)
            if f < bestf:
                bestf, best = f, t
    b = c['ybar'] - float(c['Pbar'] @ best)
    return np.concatenate([[b], best]), {'faces': nfaces, 'truncated': False,
                                         'n_support': int(np.sum(best > 1e-12))}


# ---------------------------------------------------------------- C: 真约束 trf（带 θ_i≤1 上界，无 clip）
def fit_simplex_trf(P, y):
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    A = np.column_stack([np.ones(len(y)), P[:, :K - 1] - P[:, K - 1:K]])
    rhs = y - P[:, K - 1]
    r = lsq_linear(A, rhs, bounds=(np.array([-np.inf] + [0.0] * (K - 1)),
                                   np.array([np.inf] + [1.0] * (K - 1))), method='trf')
    th = np.zeros(K); th[:K - 1] = r.x[1:]; th[K - 1] = 1.0 - th[:K - 1].sum()
    pre = float(th[:K - 1].sum())
    return np.concatenate([[r.x[0]], th]), {
        'sum_pre_clip': pre, 'clip_triggered': False,
        'bound_hit_theta_le_1': int(np.sum(np.abs(r.x[1:] - 1.0) < 1e-9)),
        'sum_pre_clip_minus1': pre - 1.0}


# ---------------------------------------------------------------- D: bvls
def fit_simplex_bvls(P, y):
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    A = np.column_stack([np.ones(len(y)), P[:, :K - 1] - P[:, K - 1:K]])
    rhs = y - P[:, K - 1]
    r = lsq_linear(A, rhs, bounds=(np.array([-np.inf] + [0.0] * (K - 1)),
                                   np.array([np.inf] + [1.0] * (K - 1))), method='bvls')
    th = np.zeros(K); th[:K - 1] = r.x[1:]; th[K - 1] = 1.0 - th[:K - 1].sum()
    pre = float(th[:K - 1].sum())
    return np.concatenate([[r.x[0]], th]), {
        'sum_pre_clip': pre, 'clip_triggered': False,
        'bound_hit_theta_le_1': int(np.sum(np.abs(r.x[1:] - 1.0) < 1e-9)),
        'sum_pre_clip_minus1': pre - 1.0}


# ---------------------------------------------------------------- E: Dirichlet 随机搜索（闭式）
def fit_simplex_dirichlet(P, y, n=N_DIR, seed=0, chunk=100_000):
    P = np.asarray(P, np.float64); y = np.asarray(y, np.float64)
    K = P.shape[1]
    c = quad_cache(P, y)
    rng = np.random.RandomState(seed)
    best, bestf = None, np.inf
    done = 0
    while done < n:
        m = min(chunk, n - done)
        TH = rng.dirichlet(np.ones(K), size=m)
        s = sse_q(c, TH)
        i = int(np.argmin(s))
        if s[i] < bestf:
            bestf = float(s[i])
            b = c['ybar'] - float(c['Pbar'] @ TH[i])
            best = np.concatenate([[b], TH[i]])
        done += m
    return best, {'random_draws': n}


SOLVERS = {
    'clip_trf':   (fit_simplex_clip,     "kairos_rule.py:40-56 fit_simplex 现行实现（变量替换 + θ_K<0 时 clip 归一）"),
    'qp_slsqp':   (fit_simplex_slsqp,    "真约束 QP: SLSQP, constraints=[Σθ=1], bounds=[(−inf,inf)]+[(0,1)]*K, 10 随机起点"),
    'qp_exact':   (fit_simplex_exact,    "真约束 QP 精确解: 枚举全部支撑面 + KKT（凸 QP 充要条件，全局最优校验）"),
    'trf_bound':  (fit_simplex_trf,      "真约束 trf: 变量替换 + bounds θ_i∈[0,1]（无 clip）"),
    'lsq_bvls':   (fit_simplex_bvls,     "lsq_linear(method='bvls') 同变量替换"),
    'dirichlet':  (fit_simplex_dirichlet, "2e5 次 Dirichlet 随机搜索 + 闭式最优截距"),
}


def main():
    rows = json.load(open(os.path.join(MID, 'results', 'matrix_4x5.json'), encoding='utf-8'))['rows']
    ktr = json.load(open(os.path.join(MID, 'results', 'kairos_theta_rule.json'), encoding='utf-8'))['cells']

    def matrix_row(ds, pr):
        for r in rows:
            if r.get('dataset') == ds and r.get('protocol') == pr:
                return r
        return None

    out = {'meta': {
        'id': 't2',
        'description': '求解器 × 单纯形约束对账：15 个有 θ 的格 × 5 条求解路径',
        'inputs': ['results/matrix_4x5.json (theta/caliber)',
                   'results/kairos_theta_rule.json (members, official/nnls_free/simplex)',
                   'results/p2/<cell>/pred_*_{val,test}.parquet'],
        'solvers': {k: v[1] for k, v in SOLVERS.items()},
        'solver_ref': 'A=clip_trf 逐字复刻 kairos_rule.py:40-56；其余四条为真约束/全局搜索对照',
        'n_dirichlet_draws': N_DIR, 'n_slsqp_starts': N_START,
        'theta_fit_caliber': '全部在 val 上拟合（θ_source=val），test 仅用于评估',
        'ci_impl': 'scripts/ci_lib.py::ci_harrell（Harrell C, tie 忽略）',
    }, 'cells': {}}

    for ds, pr, cell in CELLMAP:
        mr = matrix_row(ds, pr)
        if mr is None or not mr.get('theta'):
            print('SKIP (matrix theta 为空):', ds, pr); continue
        t0 = time.time()
        names, Pva, yva, Pte, yte, align_ok = load_cell(cell)
        ymean = float(yva.mean())
        dmax = float(np.max([abs(np.mean(Pte[:, j]) - np.mean(Pva[:, j])) / max(np.std(Pva[:, j]), 1e-9)
                             for j in range(len(names))]))
        chosen = 'simplex' if dmax > THRESH else 'nnls_free'
        single_te = {m: rmse(Pte[:, j], yte) for j, m in enumerate(names)}
        single_va = {m: rmse(Pva[:, j], yva) for j, m in enumerate(names)}
        ent = {
            'matrix_label': '%s·%s' % (ds, pr), 'p2_dir': cell,
            'members': names, 'K': len(names),
            'n_val': int(len(yva)), 'n_test': int(len(yte)), 'rows_aligned': bool(align_ok),
            'shift_d_max': round(dmax, 4), 'chosen_branch': chosen,
            'stored_theta_matrix_4x5': [float(x) for x in mr['theta']],
            'stored_caliber_matrix_4x5': mr.get('caliber'),
            'stored_official_rmse_test_kairos_theta_rule': ktr.get(cell, {}).get('official', {}).get('rmse'),
            'stored_official_theta_kairos_theta_rule': ktr.get(cell, {}).get('official', {}).get('theta'),
            'single_test': single_te, 'single_val': single_va,
            'best_single_test': min(single_te, key=lambda k: single_te[k]),
            'solvers': {},
        }
        # 参考口径：等权
        eq = np.full(len(names), 1.0 / len(names))
        b_eq = ymean - float(Pva.mean(0) @ eq)
        ent['solvers']['equal_weight_ref'] = {
            'note': '零参数基线（非 5 个求解器之一）',
            'theta': [float(x) for x in eq], 'intercept': float(b_eq), 'sum_theta': 1.0,
            'val_rmse': rmse(b_eq + Pva @ eq, yva), 'test_rmse': rmse(b_eq + Pte @ eq, yte),
            'test_ci': float(ci_harrell(b_eq + Pte @ eq, yte)),
            'sum_theta_before_clip': 1.0, 'clip_triggered': False, 'fit_seconds': 0.0,
        }
        # 参考口径：nnls_free（现行 NNLS，θ≥0 截距自由）
        th_n = fit_nnls(Pva, yva)
        ent['solvers']['nnls_free_ref'] = {
            'note': "零参数外的现行 free-intercept NNLS（非 5 个求解器之一）",
            'theta': [float(x) for x in th_n[1:]], 'intercept': float(th_n[0]),
            'sum_theta': float(np.sum(th_n[1:])),
            'val_rmse': rmse(apply_th(th_n, Pva), yva), 'test_rmse': rmse(apply_th(th_n, Pte), yte),
            'test_ci': float(ci_harrell(apply_th(th_n, Pte), yte)),
            'sum_theta_before_clip': None, 'clip_triggered': False, 'fit_seconds': None,
        }
        # 5 条求解路径
        cache = quad_cache(Pva, yva)
        for sname, (fn, _) in SOLVERS.items():
            ts = time.time()
            th, diag = fn(Pva, yva)
            el = time.time() - ts
            pva, pte = apply_th(th, Pva), apply_th(th, Pte)
            sse_opt = float(sse_q(cache, th[1:][None, :])[0]) / cache['N']
            ent['solvers'][sname] = {
                'theta': [float(x) for x in th[1:]], 'intercept': float(th[0]),
                'sum_theta': float(np.sum(th[1:])),
                'sum_theta_before_clip': (None if diag.get('sum_pre_clip') is None
                                          else float(diag['sum_pre_clip'])),
                'clip_triggered': bool(diag.get('clip_triggered', False)),
                'bound_hit_theta_le_1': int(diag.get('bound_hit_theta_le_1', 0)),
                'theta_min': float(np.min(th[1:])), 'theta_max': float(np.max(th[1:])),
                'simplex_feasible': bool(float(np.min(th[1:])) >= -1e-9
                                          and abs(float(np.sum(th[1:])) - 1.0) <= 1e-9),
                'val_rmse': rmse(pva, yva), 'val_mse_closed_form': sse_opt,
                'test_rmse': rmse(pte, yte), 'test_ci': float(ci_harrell(pte, yte)),
                'fit_seconds': round(el, 3),
                'extra': {k: v for k, v in diag.items()
                          if k not in ('sum_pre_clip', 'clip_triggered', 'bound_hit_theta_le_1')},
            }
        # 全局最优性：仅在**可行**（真在单纯形上）的求解器之间比较 val MSE
        vm = {k: v['val_rmse'] ** 2 for k, v in ent['solvers'].items() if v.get('val_rmse') is not None}
        ev = ent['solvers']['qp_exact']['val_rmse'] ** 2
        ent['exact_simplex_val_mse'] = ev
        ent['val_mse_gap_vs_exact'] = {k: round(vm[k] - ev, 12) for k in vm}
        ent['solver_reaches_exact'] = {k: bool(vm[k] - ev <= 1e-10) for k in vm}
        # 只有真正落在单纯形上的求解器参与"最优单纯形解"的比较；
        # equal_weight_ref / nnls_free_ref 是参照（后者根本不在单纯形上：Σθ≠1）。
        feas = [k for k in SOLVERS if ent['solvers'][k].get('simplex_feasible')]
        ent['simplex_feasible_solvers'] = feas
        ent['simplex_infeasible_solvers'] = [k for k in SOLVERS if k not in feas]
        best_key = min(feas, key=lambda k: vm[k])
        ent['best_simplex_solver'] = best_key
        ent['best_simplex_val_rmse'] = ent['solvers'][best_key]['val_rmse']
        ent['best_val_solver'] = best_key
        ent['val_mse_gap_vs_best'] = {k: round(vm[k] - vm[best_key], 12) for k in vm}
        ent['total_seconds'] = round(time.time() - t0, 2)
        out['cells']['%s::%s' % (cell, ent['matrix_label'])] = ent

        c1 = ent['solvers']['clip_trf']; c2 = ent['solvers']['qp_slsqp']
        c3 = ent['solvers']['qp_exact']
        print('%-16s K=%2d %-10s | clip test %.6f Σθ=%.4f %s feas=%s | SLSQP test %.6f feas=%s | EXACT test %.6f | Δval(clip-exact) %.6f | best_feasible %s'
              % (cell, ent['K'], chosen, c1['test_rmse'], c1['sum_theta'],
                 'CLIP' if c1['clip_triggered'] else 'clean', c1['simplex_feasible'],
                 c2['test_rmse'], c2['simplex_feasible'], c3['test_rmse'],
                 c1['val_rmse'] - c3['val_rmse'], best_key))

    json.dump(out, open(os.path.join(OUT, 'solver_matrix.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print('\n写入', os.path.join(OUT, 'solver_matrix.json'))
    print('格数 =', len(out['cells']))


if __name__ == '__main__':
    main()