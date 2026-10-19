# -*- coding: utf-8 -*-
"""t5-(3) KIBA 非 pK 尺度审计 + 逐格地板倍数与适用范围。
输出 D:/protein_datasets/midas/results/strengthen/t5/kiba_scale.json
"""
import os, json
import numpy as np
import pandas as pd

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(MID, 'results', 'strengthen', 't5')
os.makedirs(OUT, exist_ok=True)

FLOOR = 0.5108  # results/exp_gap/q/q11.json：同 assay 重复测量 pooled SD（ChEMBL log10 单位）

DS = {
    'davis': 'data/splits/davis_prepared.parquet',
    'kiba': 'data/splits/kiba_prepared.parquet',
    'chembl': 'data/splits/chembl_prepared.parquet',
    'bindingdb': 'data/splits/bindingdb_prepared.parquet',
}


def quant(ds, path, src_note):
    df = pd.read_parquet(os.path.join(MID, path), columns=['pK'])
    v = df['pK'].values.astype(float)
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        'source': src_note,
        'n': int(len(v)),
        'min': float(v.min()), 'q01': float(q[1]), 'q05': float(q[2]),
        'q25': float(q[3]), 'median': float(q[4]), 'q75': float(q[5]),
        'q95': float(q[6]), 'q99': float(q[7]), 'max': float(v.max()),
        'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
        'frac_gt_10': float((v > 10).mean()),
        'frac_gt_11': float((v > 11).mean()),
        'frac_ge_11_72': float((v >= 11.72).mean()),
        'frac_eq_5': float((v == 5.0).mean()),
        # 中心化后跨度（离散度），便于跨数据集比较"量纲"
        'unit_span_needed_for_typical_pair_diff': dict(
            q25_q75_iqr=float(q[5] - q[3]),
            sd=float(v.std(ddof=1))),
    }


scale = {'floor_definition': {
    'value': FLOOR,
    'source': 'results/exp_gap/q/q11.json',
    'definition': '同 assay 同 (化合物,靶点) 重复测量的 pooled SD，单位 = log10 (pK / M)',
    'domain_restriction': '仅对 log10 单位的亲和力标签（Davis pK / ChEMBL pK / BindingDB pK）有意义，不能与 kcal/mol 或任意非 log 分数并列',
    'provenance_caveat': '严格说 0.5108 是在 ChEMBL 上估计的同 assay 重复地板；把它当分母套到 BindingDB / 激酶基准 / Davis 是跨库近似（Davis 无 assay 分层字段，无法在本库内独立估计）。本文件对 Davis/BindingDB/kinase 三组仍标 applicable=true 并同时给出该 caveat；对 KIBA 标 applicable=false（量纲不同，非近似问题）。',
}, 'distributions': {}}
for ds, path in DS.items():
    scale['distributions'][ds] = quant(ds, path, path)

# ---- KIBA 与 pK 数据集的偏移（复核 kiba_caliber.json 的 +6.302 是哪一种口径）----
kiba = pd.read_parquet(os.path.join(MID, DS['kiba']), columns=['pK'])['pK'].values.astype(float)
bdb = pd.read_parquet(os.path.join(MID, DS['bindingdb']), columns=['pK'])['pK'].values.astype(float)
csn = json.load(open(os.path.join(MID, 'results', 'cross_source_noise.json'), encoding='utf-8'))
scale['kiba_vs_bindingdb_offset'] = {
    'marginal_mean_kiba': float(kiba.mean()), 'marginal_mean_bindingdb': float(bdb.mean()),
    'marginal_mean_offset': float(kiba.mean() - bdb.mean()),
    'matched_pair_n': csn['kiba_vs_bindingdb']['n_exact'],
    'matched_pair_mean_diff_kiba_minus_bdb': csn['kiba_vs_bindingdb']['mean_diff'],
    'matched_pair_sd': csn['kiba_vs_bindingdb']['sd_diff'],
    'kiba_caliber_json_claim': 6.302,
    'claim_is_matched_pair_offset': bool(abs(csn['kiba_vs_bindingdb']['mean_diff'] - 6.302) < 0.05),
    'claim_is_marginal_offset': bool(abs((kiba.mean() - bdb.mean()) - 6.302) < 0.05),
    'reproduced_matched_pair_offset': round(float(csn['kiba_vs_bindingdb']['mean_diff']), 4),
    'source_of_matched_offset': 'results/cross_source_noise.json → kiba_vs_bindingdb.mean_diff（1,885 对同 (化合物,靶点) 匹配）',
    'interpretation': 'KIBA 是「KIBA score」合成分数（kinase-inhibitor 融合 Kd/Ki/IC50 的归一化分值），不是 −log10(K) 的 pK。所谓 +6.302 是 1,885 对匹配化合物的同对均值差，不是两库边际均值差（后者 %.4f）。把 log10 地板 0.5108 相除是量纲错误。' % (kiba.mean() - bdb.mean()),
}
# KIBA 与 Davis 的同量纲检验：Davis pK 的单位离散度 vs KIBA
dav = pd.read_parquet(os.path.join(MID, DS['davis']), columns=['pK'])['pK'].values.astype(float)
scale['dimensionality_check'] = {
    'davis_sd': float(dav.std(ddof=1)), 'kiba_sd': float(kiba.std(ddof=1)),
    'chembl_sd': float(pd.read_parquet(os.path.join(MID, DS['chembl']), columns=['pK'])['pK'].values.astype(float).std(ddof=1)),
    'bindingdb_sd': float(bdb.std(ddof=1)),
    'note': 'KIBA 与 pK 数据集的 sd 同量级，但量纲基准不同（KIBA min=0 附近、pK min=5.0）；分母 0.5108 只对 pK 类有效。',
    'kiba_min': float(kiba.min()), 'kiba_has_negative': bool((kiba < 0).any()),
}

# ---- 逐格地板倍数：从 matrix_4x5.json + floor，按数据集分组给适用范围 ----
matrix = json.load(open(os.path.join(MID, 'results', 'matrix_4x5.json'), encoding='utf-8'))
q11 = json.load(open(os.path.join(MID, 'results', 'exp_gap', 'q', 'q11.json'), encoding='utf-8'))

DS_OF = {
    'ChEMBL 战场': 'chembl', 'BindingDB': 'bindingdb', 'Davis': 'davis',
    'KIBA': 'kiba', '激酶基准': 'kinase',
}
PROT_OF = {'时间': 'temporal', '随机': 'random', 'scaffold': 'scaffold', '冷启动': 'cold'}
FLOOR_APPLICABLE = {'chembl': True, 'bindingdb': True, 'davis': True, 'kinase': True, 'kiba': False}
# q11.json 用 kiba_official 指代 KIBA 官方 setting1 随机格；与该 json 对齐以免缺格
CELL_KEY_OVERRIDE = {('kiba', 'random'): 'kiba_official'}

cells = {}
for r in matrix['rows']:
    if r.get('status') != 'DONE':
        continue
    ds = DS_OF[r['dataset']]
    key = '%s_%s' % (ds, PROT_OF[r['protocol']])
    qkey = CELL_KEY_OVERRIDE.get((ds, PROT_OF[r['protocol']]), key)
    ours = r['ours']['rmse']
    opp = r['opponent']['rmse']
    q = q11['cells'].get(qkey)
    applicable = FLOOR_APPLICABLE[ds]
    cells[key] = {
        'dataset': ds, 'protocol': PROT_OF[r['protocol']],
        'q11_cell_name': qkey,
        'official_rmse': ours, 'opponent': r['opponent']['name'], 'opponent_rmse': opp,
        'official_over_floor': round(ours / FLOOR, 4),
        'opponent_over_floor': round(opp / FLOOR, 4),
        'floor_applicable': applicable,
        'floor_in_same_library': ds == 'chembl',
        'floor_same_library_note': ('同库估计（ChEMBL 内同 assay 重复）' if ds == 'chembl'
                                    else '跨库近似：0.5108 是 ChEMBL 同 assay 地板，本数据集无 assay 分层字段可独立估计'
                                    if applicable else 'n/a（量纲不同）'),
        'floor_label': ('ok (log10 pK units)' if applicable else 'floor not applicable (non-log score)'),
        'q11_present': q is not None,
        'q11_official_over_floor': (q['official_over_floor'] if q else None),
        'q11_opponent_over_floor': (q['opponent_over_floor'] if q else None),
        'q11_consistent': (None if not q else bool(abs(q['official_over_floor'] - ours / FLOOR) < 5e-3
                                                   and abs(q['opponent_over_floor'] - opp / FLOOR) < 5e-3)),
    }
scale['per_cell_floor_multiples'] = cells
appl = [c['official_over_floor'] for c in cells.values() if c['floor_applicable']]
allm = [c['official_over_floor'] for c in cells.values()]
scale['floor_multiple_ranges'] = {
    'all_18_cells': {'min': float(min(allm)), 'max': float(max(allm)), 'median': float(np.median(allm)),
                     'min_cell': min(cells, key=lambda k: cells[k]['official_over_floor']),
                     'max_cell': max(cells, key=lambda k: cells[k]['official_over_floor'])},
    'floor_applicable_only': {'n': len(appl), 'min': float(min(appl)), 'max': float(max(appl)),
                              'median': float(np.median(appl)),
                              'min_cell': min([k for k in cells if cells[k]['floor_applicable']],
                                              key=lambda k: cells[k]['official_over_floor']),
                              'max_cell': max([k for k in cells if cells[k]['floor_applicable']],
                                              key=lambda k: cells[k]['official_over_floor'])},
    'kiba_cells_excluded': [k for k in cells if not cells[k]['floor_applicable']],
}
json.dump(scale, open(os.path.join(OUT, 'kiba_scale.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(scale, ensure_ascii=False, indent=1))
print('\nWROTE kiba_scale.json')