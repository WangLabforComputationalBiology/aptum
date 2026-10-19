# -*- coding: utf-8 -*-
"""
t2 · Θ→成员显式映射表 + 映射唯一性自证 + clip 分支正反两版重算 official
=====================================================================
只读输入：results/matrix_4x5.json, results/kairos_theta_rule.json,
          results/p2/<cell>/pred_*_{val,test}.parquet, scripts/kairos_rule.py
新产物只写 results/strengthen/t2/：
  theta_member_map.csv                      格 × 10 成员 × 权重正负号 × 单成员 RMSE
  mapping_uniqueness.json                   每格：落盘映射重建 RMSE + 全部权重排列的重建 RMSE 分布
  kairos_theta_rule_official_qp.json        仅 official 分支：clip 路径 与 纯 QP 两版并列（冻结文件不改）
  three_numbers_provenance.json             1.0846/1.0431/1.0484/0.0808 逐条来源
"""
import os, sys, json, csv, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kairos_rule import fit_nnls, fit_simplex, apply_th, THRESH
from ci_lib import ci_harrell
from t2_solvers import load_cell, fit_simplex_slsqp, fit_simplex_clip, rmse

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(MID, 'results', 'strengthen', 't2')
os.makedirs(OUT, exist_ok=True)

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


def main():
    rows = json.load(open(os.path.join(MID, 'results', 'matrix_4x5.json'), encoding='utf-8'))['rows']
    ktr = json.load(open(os.path.join(MID, 'results', 'kairos_theta_rule.json'), encoding='utf-8'))['cells']

    def mrow(ds, pr):
        for r in rows:
            if r.get('dataset') == ds and r.get('protocol') == pr:
                return r
        return None

    map_rows = []
    uniq = {}
    official_qp = {'meta': {
        'description': "重算 kairos_theta_rule.json 的 official 分支：clip 路径 vs 纯 QP（真约束），"
                       "冻结文件未改动，仅在本目录写副本",
        'clip_path': 'scripts/kairos_rule.py:40-56 fit_simplex（变量替换 + 第 54-55 行 clip 归一）',
        'qp_path': "scipy.optimize.minimize(SLSQP, constraints=[Σθ=1], bounds=[(−inf,inf)]+[(0,1)]*K, 10 Dirichlet 起点)",
        'source_frozen': 'results/kairos_theta_rule.json',
    }, 'cells': {}}

    for ds, pr, cell in CELLMAP:
        mr = mrow(ds, pr)
        if mr is None or not mr.get('theta'):
            continue
        names, Pva, yva, Pte, yte, ok = load_cell(cell)
        th_stored = np.asarray(mr['theta'], float)
        # 落盘映射重建
        rebuild_te_stored = rmse(th_stored[0] + Pte @ th_stored[1:], yte)
        rebuild_va_stored = rmse(th_stored[0] + Pva @ th_stored[1:], yva)
        kairos_key = cell if cell != 'kiba_random' else 'kiba_random'
        off = ktr.get(cell, {}).get('official', {})
        # 现行 clip 路径（逐字复刻）与纯 QP
        th_clip, d_clip = fit_simplex_clip(Pva, yva)
        th_qp, d_qp = fit_simplex_slsqp(Pva, yva)
        dmax = float(np.max([abs(np.mean(Pte[:, j]) - np.mean(Pva[:, j])) / max(np.std(Pva[:, j]), 1e-9)
                             for j in range(len(names))]))
        chosen = 'simplex' if dmax > THRESH else 'nnls_free'
        official_qp['cells'][cell] = {
            'chosen': chosen, 'shift_d_max': round(dmax, 4),
            'members': names,
            'frozen_official_theta': off.get('theta'),
            'frozen_official_rmse_test': off.get('rmse'),
            'frozen_official_rmse_test_from_mse_field': (float(np.sqrt(off['mse'])) if off.get('mse') else None),
            'clip_path': {
                'theta': [float(x) for x in th_clip[1:]], 'intercept': float(th_clip[0]),
                'sum_theta_before_clip': float(d_clip['sum_pre_clip']),
                'clip_triggered': bool(d_clip['clip_triggered']),
                'val_rmse': rmse(apply_th(th_clip, Pva), yva),
                'test_rmse': rmse(apply_th(th_clip, Pte), yte),
                'test_ci': float(ci_harrell(apply_th(th_clip, Pte), yte)),
            },
            'qp_path': {
                'theta': [float(x) for x in th_qp[1:]], 'intercept': float(th_qp[0]),
                'sum_theta_before_clip': None, 'clip_triggered': False,
                'val_rmse': rmse(apply_th(th_qp, Pva), yva),
                'test_rmse': rmse(apply_th(th_qp, Pte), yte),
                'test_ci': float(ci_harrell(apply_th(th_qp, Pte), yte)),
            },
        }
        # 映射唯一性：枚举非零权重在成员间的全部排列（截距固定）
        nz = [(i, th_stored[1 + i]) for i in range(len(names)) if abs(th_stored[1 + i]) > 1e-12]
        Z = [(i, float(v)) for i, v in nz]
        rebuilds = []
        if len(Z) <= 8:
            idxs = [z[0] for z in Z]; vals = [z[1] for z in Z]
            for perm in set(itertools.permutations(range(len(vals)))):
                w = np.zeros(len(names))
                for slot, src in zip(idxs, perm):
                    w[slot] = vals[src]
                rebuilds.append({'assign': {names[s]: round(vals[src], 6) for s, src in zip(idxs, perm)},
                                 'test_rmse': rmse(th_stored[0] + Pte @ w, yte)})
        rebuilds_sorted = sorted(rebuilds, key=lambda z: z['test_rmse'])
        # 落盘（按名归位）映射的排名
        stored_map = {names[i]: round(float(th_stored[1 + i]), 6) for i in idxs}
        stored_rank = next((k for k, z in enumerate(rebuilds_sorted)
                            if z['assign'] == stored_map), None)
        uniq[cell] = {
            'members_sorted_order': names, 'stored_theta': [float(x) for x in th_stored],
            'intercept': float(th_stored[0]),
            'nonzero_weight_members': stored_map,
            'rebuild_test_rmse_stored_mapping': rebuild_te_stored,
            'rebuild_val_rmse_stored_mapping': rebuild_va_stored,
            'n_weight_permutations': len(rebuilds),
            'stored_mapping_rank_by_test_rmse': stored_rank,
            'best_permutation': rebuilds_sorted[0] if rebuilds_sorted else None,
            'worst_permutation': rebuilds_sorted[-1] if rebuilds_sorted else None,
            'runner_up': rebuilds_sorted[1] if len(rebuilds_sorted) > 1 else None,
        }
        for i, nm in enumerate(names):
            map_rows.append({
                'cell': cell, 'matrix_label': '%s·%s' % (ds, pr), 'member_index_sorted': i,
                'member': nm, 'theta_matrix_4x5': round(float(th_stored[1 + i]), 6),
                'theta_sign': ('+' if th_stored[1 + i] > 1e-12 else
                               ('-' if th_stored[1 + i] < -1e-12 else '0')),
                'is_intercept_separate': True,
                'intercept': round(float(th_stored[0]), 6),
                'sum_member_theta': round(float(np.sum(th_stored[1:])), 6),
                'member_val_rmse': round(rmse(Pva[:, i], yva), 6),
                'member_test_rmse': round(rmse(Pte[:, i], yte), 6),
                'stored_official_theta_kairos': (off.get('theta') or [None] * (len(names) + 1))[1 + i],
            })
        print('%-18s stored-map rebuild test %.6f (rank %s/%d) | clip=%s | QP test %.6f'
              % (cell, rebuild_te_stored, stored_rank, len(rebuilds),
                 d_clip['clip_triggered'], official_qp['cells'][cell]['qp_path']['test_rmse']))

    with open(os.path.join(OUT, 'theta_member_map.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
        wr.writeheader(); wr.writerows(map_rows)
    json.dump(uniq, open(os.path.join(OUT, 'mapping_uniqueness.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    json.dump(official_qp, open(os.path.join(OUT, 'kairos_theta_rule_official_qp.json'), 'w',
                                encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\n写入 theta_member_map.csv / mapping_uniqueness.json / kairos_theta_rule_official_qp.json')


if __name__ == '__main__':
    main()