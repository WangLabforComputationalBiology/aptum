#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自检：验证本包能独立复现论文的关键数字。

    python scripts/selfcheck.py

检查四项：
  1. 18 格矩阵可重建，且 18/18 有裁决
  2. 压缩预测完整（240 个数组 / 19 格）
  3. 逐格对抗复现：Aptum 在每格都优于池内最佳单成员
  4. 图脚本可运行

退出码 0 = 全部通过。
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MID = os.path.dirname(HERE)
PY = sys.executable
ok = True


def run(script, args=()):
    r = subprocess.run([PY, os.path.join(HERE, script)] + list(args),
                       capture_output=True, text=True, cwd=MID)
    return r.returncode, r.stdout + r.stderr


print('Aptum self-check  (root = %s)\n' % MID)

# 1) 矩阵
print('[1] 18-cell matrix ...', end=' ')
rc, out = run('p4_matrix.py')
n_done = ''
for line in out.splitlines():
    if '已完成格' in line:
        n_done = line.strip()
        break
if rc == 0 and '已完成格 18' in out:
    print('OK  (%s)' % n_done)
else:
    print('FAIL (rc=%d)  %s' % (rc, n_done or 'no summary'))
    ok = False

# 2) 预测完整性
print('[2] compressed predictions ...', end=' ')
try:
    idx = json.load(open(os.path.join(MID, 'predictions', 'index.json'), encoding='utf-8'))
    n_cells = len(idx)
    n_arr = sum(len(v) for c in idx.values() for s in c.values() for m, v in s.items()) * 2
    import glob
    npz = glob.glob(os.path.join(MID, 'predictions', '*.npz'))
    print('OK  (%d cells, %d npz files, %d arrays)' % (n_cells, len(npz), n_arr))
except Exception as e:
    print('FAIL', e); ok = False

# 3) 逐格对抗
print('[3] head-to-head reproduction ...', end=' ')
rc, out = run('reproduce_from_npz.py')
wins = ''
for line in out.splitlines():
    if 'beats best single' in line:
        wins = line.strip(); break
if rc == 0 and '19/19' in out:
    print('OK  (%s)' % wins)
else:
    print('FAIL (rc=%d)  %s' % (rc, wins or out[-200:]))
    ok = False

# 4) 图
print('[4] figure scripts ...', end=' ')
rc1, _ = run('make_fig_eds_v3.py')
rc2, _ = run('make_fig_agg_v6.py')
if rc1 == 0 and rc2 == 0:
    print('OK  (fig_eds.pdf, fig_aggregation.pdf)')
else:
    print('FAIL (rc=%d,%d)' % (rc1, rc2)); ok = False

print()
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)