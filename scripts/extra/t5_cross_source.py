# -*- coding: utf-8 -*-
"""t5-(2) 跨源噪声「去删失」复算。
复现 results/cross_source_noise_davis_bdb.json 的 294 对独立测量，
并按 Davis 侧 pK 是否 > 5.0（= 是否删失）切子集重算 sd / median|Δ| / q90|Δ|。
原始配对文件已定位：sota_duel/ext/davis_pairs.parquet + D:/protein_datasets/bindingdb/bdb_pairs.parquet
（与 scripts/e4b_refine.py 完全同一路径与口径）。
输出 D:/protein_datasets/midas/results/strengthen/t5/cross_source_decensored.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(MID, 'results', 'strengthen', 't5')
EXT = os.path.join(MID, 'sota_duel', 'ext')
BDB = 'D:/protein_datasets/bindingdb/bdb_pairs.parquet'
FROZEN = json.load(open(os.path.join(MID, 'results', 'cross_source_noise_davis_bdb.json'), encoding='utf-8'))
_cache = {}


def can(s):
    if s not in _cache:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        _cache[s] = Chem.MolToSmiles(m) if m is not None else None
    return _cache[s]


def prep(df, sc, qc, pc):
    m = {s: can(s) for s in df[sc].dropna().unique()}
    d = df[[sc, qc, pc]].copy()
    d['csmiles'] = d[sc].map(m)
    d = d.dropna(subset=['csmiles', qc, pc])
    return d.groupby(['csmiles', qc], as_index=False)[pc].median().rename(columns={qc: 'seq', pc: 'pK'})


davis = prep(pd.read_parquet(os.path.join(EXT, 'davis_pairs.parquet')), 'smiles', 'sequence', 'pK')
bdb = prep(pd.read_parquet(BDB), 'smiles', 'seq', 'pK')
j = davis.merge(bdb, on=['csmiles', 'seq'], suffixes=('_d', '_b'))
j['delta'] = j['pK_d'].values - j['pK_b'].values
same = np.abs(j['delta'].values) < 1e-6
ind = j[~same].copy()


def stats(v):
    v = np.asarray(v, float)
    a = np.abs(v)
    return {'n': int(len(v)), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None,
            'rmse': float(np.sqrt((v ** 2).mean())),
            'median_abs': float(np.median(a)), 'q25_abs': float(np.quantile(a, .25)),
            'q75_abs': float(np.quantile(a, .75)), 'q90_abs': float(np.quantile(a, .90)),
            'frac_abs_gt_1': float((a > 1).mean())}


res = {
    'raw_files_located': True,
    'davis_pairs': 'sota_duel/ext/davis_pairs.parquet',
    'bindingdb_pairs': 'D:/protein_datasets/bindingdb/bdb_pairs.parquet',
    'protocol': '与 scripts/e4b_refine.py 同口径：canonical SMILES + 靶点序列精确匹配，组内取 median',
    'reproduction_check': {
        'n_matched': int(len(j)), 'frozen_n_matched': FROZEN['n_matched'],
        'n_identical_record': int(same.sum()), 'frozen_n_identical_record': FROZEN['n_identical_record'],
        'n_independent_measurement': int(len(ind)), 'frozen_n_independent_measurement': FROZEN['n_independent_measurement'],
        'reproduces_frozen': bool(len(j) == FROZEN['n_matched']
                                  and int(same.sum()) == FROZEN['n_identical_record']
                                  and len(ind) == FROZEN['n_independent_measurement']),
    },
    'with_censoring': {'definition': '全部 294 对独立测量（含 Davis 侧 pK == 5.0 的删失对）',
                       **stats(ind['delta'].values)},
    'decensored': {'definition': 'Davis 侧 pK > 5.0 的独立测量对（剔除批底删失）',
                   **stats(ind[ind.pK_d > 5.0]['delta'].values)},
    'censored_side_only': {'definition': 'Davis 侧 pK == 5.0 的独立测量对',
                           **stats(ind[ind.pK_d == 5.0]['delta'].values)},
    'frozen_json_reference': FROZEN['independent_only'],
}
res['decensored_effect'] = {
    'sd_with': res['with_censoring']['sd'], 'sd_without': res['decensored']['sd'],
    'sd_change': res['decensored']['sd'] - res['with_censoring']['sd'],
    'median_abs_with': res['with_censoring']['median_abs'],
    'median_abs_without': res['decensored']['median_abs'],
    'n_dropped': res['with_censoring']['n'] - res['decensored']['n'],
}
json.dump(res, open(os.path.join(OUT, 'cross_source_decensored.json'), 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
print('\nWROTE cross_source_decensored.json')