# -*- coding: utf-8 -*-
"""从压缩后的 predictions/*.npz 重算 18 格矩阵口径下的关键数字。
这是论文 "without retraining" 复现路径的入口脚本。

用法：
    python scripts/reproduce_from_npz.py

依赖：numpy, scipy, pandas（不需要 GPU、不需要原始数据集）
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MID = os.path.dirname(HERE)
PRED = os.path.join(MID, 'predictions')
sys.path.insert(0, HERE)
from kairos_rule import fit_simplex, fit_nnls        # noqa: E402
from ci_lib import ci_harrell                        # noqa: E402


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def load_cell(cell):
    """读一格的全部成员预测。

    格式（去重后）：每个 split 只存一份标签
        <cell>__<split>__pK                  标签
        <cell>|<member>__<split>__pred       成员预测
    """
    fp = os.path.join(PRED, '%s.npz' % cell.replace(' ', '_'))
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    out = {}
    for k in z.files:
        if k.endswith('__pK'):
            base_, split, _ = k.split('__')
            out.setdefault(split, {}).setdefault('__label__', z[k].astype(np.float32))
        elif k.endswith('__pred'):
            base_, split, _ = k.split('__')
            cell_, mem = base_.split('|')
            out.setdefault(split, {}).setdefault(mem, {})['pred'] = z[k].astype(np.float32)
    return out


def cv5(X, y, nf=5, seed=0):
    n = len(y); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); p = np.zeros(n)
    for f in np.array_split(idx, nf):
        t = np.setdiff1d(np.arange(n), f)
        th = fit_simplex(X[t], y[t]); p[f] = X[f] @ th[1:] + th[0]
    return p


def main():
    idx_path = os.path.join(PRED, 'index.json')
    if not os.path.exists(idx_path):
        print('predictions/index.json not found. 先运行 predictions/ 的打包脚本。')
        return 1
    index = json.load(open(idx_path, encoding='utf-8'))

    rows = []
    print('%-20s %4s %7s | %10s %8s | %10s %8s' % (
        'cell', 'K', 'n_test', 'Aptum', 'CI', 'best single', 'ratio'))
    print('-' * 78)
    for cell in sorted(index):
        data = load_cell(cell)
        if data is None or 'test' not in data:
            continue
        mems = sorted(m for m in data['test'] if m != '__label__' and 'kairos' not in m.lower())
        if len(mems) < 2:
            continue
        Xt = np.column_stack([data['test'][m]['pred'] for m in mems])
        yt = data['test']['__label__']
        ok = np.isfinite(Xt).all(1) & np.isfinite(yt)
        Xt, yt = Xt[ok], yt[ok]
        if len(yt) < 50:
            continue
        p_ap = cv5(Xt, yt)
        r_ap = rmse(p_ap, yt); c_ap = float(ci_harrell(p_ap, yt))
        singles = {m: rmse(Xt[:, i], yt) for i, m in enumerate(mems)}
        bm = min(singles, key=singles.get)
        r_bs = singles[bm]
        rows.append(dict(cell=cell, K=len(mems), n=int(len(yt)),
                         aptum=r_ap, ci=c_ap, best_single=r_bs, best_name=bm,
                         ratio=r_bs / r_ap))
        print('%-20s %4d %7d | %10.4f %8.4f | %10.4f %8.3f' % (
            cell, len(mems), len(yt), r_ap, c_ap, r_bs, r_bs / r_ap))

    if rows:
        wins = sum(1 for r in rows if r['aptum'] < r['best_single'])
        print('-' * 78)
        print('cells=%d  RMSE beats best single member: %d/%d' % (len(rows), wins, len(rows)))
        print('mean reduction vs best single: %+.2f%%' % (
            100 * np.mean([(r['best_single'] - r['aptum']) / r['best_single'] for r in rows])))
        json.dump(rows, open(os.path.join(MID, 'results', 'reproduced_from_npz.json'), 'w'),
                  indent=1)
        print('\nwritten results/reproduced_from_npz.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())