# -*- coding: utf-8 -*-
"""
CI (Harrell's concordance, ties ignored) 的正确实现
==================================================
为什么需要这个文件
------------------
此前全项目使用的是"分块计数"版：

    o = np.argsort(p); ys = y[o]; B = 250
    for i in range(0, len(ys), B):
        yi = ys[i:i+B][:, None]
        for j in range(i+B, len(ys), B):
            yj = ys[j:j+B][None, :]
            conc += (yi < yj).sum(); disc += (yi > yj).sum()

它只统计**跨块**配对，完全漏掉块内配对，因此：
  * n <= 250 时 conc=disc=0 → 直接返回 nan（n=10 的完美预测也返回 nan）
  * n 远大于 250 时系统性**高估**（只保留了相隔较远的配对，而这些在预测序上更一致）
实测偏差：n=300 时 0.9461 vs 真值 0.8448（+0.1013）；n=2652 时约 +0.09。

正确口径（Harrell's C，tie 忽略）
--------------------------------
按预测值 p 排序后：
  * 令 disc = 在 p 序下 y 的**严格逆序对**数（p_i < p_j 且 y_i > y_j）
  * 令 comp = 全部"可比对"数 = C(n,2) − (y 相等的对数) − (p 相等的对数)
  * CI = 1 − disc / comp        （因为 comp = conc + disc）

实现用 Fenwick 树按 p 分组处理，O(n log n)，并保证 p 相等的对不计入可比对。
"""

import numpy as np


def ci_harrell(p, y):
    """Harrell's concordance index，tie 忽略（y 相等或 p 相等的对都不计）。

    与 lifelines.utils.concordance_index 口径一致（针对"越大越好"的方向）。
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        return float('nan')

    # --- 可比对数 = C(n,2) - (y并列对) - (p并列对) + (y并列且p并列的对) ---
    def _tie_pairs(v):
        _, c = np.unique(v, return_counts=True)
        return int(np.sum(c * (c - 1) // 2))

    total = n * (n - 1) // 2
    ty, tp = _tie_pairs(y), _tie_pairs(p)

    # --- 逆序对（按 p 升序，y 严格下降），Fenwick 树，按 p 分组以排除 p 并列 ---
    order = np.lexsort((y, p))          # 主键 p 升序，次键 y 升序
    ys = y[order]
    ps = p[order]
    uniq_y, inv_y = np.unique(ys, return_inverse=True)
    m = len(uniq_y)

    tree = np.zeros(m + 1, dtype=np.int64)

    def _upd(i):
        i += 1
        while i <= m:
            tree[i] += 1
            i += i & (-i)

    def _qry(i):
        """已插入元素中 rank <= i 的个数"""
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    disc = 0
    seen = 0
    i = 0
    while i < n:
        j = i
        while j < n and ps[j] == ps[i]:      # 同一 p 值的整组
            j += 1
        # 组内先查（此时树里只有严格更小 p 的元素），再统一插入
        for k in range(i, j):
            r = inv_y[k]
            disc += seen - _qry(r)            # 已有元素中 y 严格更大的个数
        for k in range(i, j):
            _upd(inv_y[k])
        seen += j - i
        i = j

    # p 并列且 y 并列的对被减了两次，加回来。
    # 注意：不要用 np.unique(..., axis=0) —— 在 15 万行上每次调用都很贵，
    # 而 bootstrap 要调数百次。改用 lexsort + run-length，纯 numpy 向量化。
    tp_and_ty = 0
    if tp and ty:
        o2 = np.lexsort((y, p))
        ps2, ys2 = p[o2], y[o2]
        same = (ps2[1:] == ps2[:-1]) & (ys2[1:] == ys2[:-1])
        if same.any():
            # run 边界处的长度
            idx = np.flatnonzero(~same)
            starts = np.concatenate(([0], idx + 1))
            ends = np.concatenate((idx + 1, [n]))
            cl = ends - starts
            tp_and_ty = int(np.sum(cl * (cl - 1) // 2))
    comp = total - ty - tp + tp_and_ty
    if comp <= 0:
        return float('nan')
    return 1.0 - disc / comp


def ci_bruteforce(p, y):
    """O(n^2) 直接枚举，仅用于小规模自检。"""
    p = np.asarray(p, float); y = np.asarray(y, float)
    conc = disc = 0
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] == p[j] or y[i] == y[j]:
                continue
            if (p[i] < p[j]) == (y[i] < y[j]):
                conc += 1
            else:
                disc += 1
    if conc + disc == 0:
        return float('nan')
    return conc / (conc + disc)


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    print('=== 与 O(n^2) 暴力枚举对照 ===')
    for n in [10, 31, 99, 300, 501, 1500]:
        y = rng.randn(n)
        p = y + 0.6 * rng.randn(n)
        a, b = ci_harrell(p, y), ci_bruteforce(p, y)
        ok = abs(a - b) < 1e-12
        print('  n=%-5d fast=%.10f brute=%.10f %s' % (n, a, b, '✓' if ok else '✗ 差 %.2e' % abs(a - b)))

    print('\n=== 含并列（y 与 p 都有 ties）的对照 ===')
    for n in [60, 400]:
        y = rng.randint(0, 5, n).astype(float)
        p = rng.randint(0, 7, n).astype(float)
        a, b = ci_harrell(p, y), ci_bruteforce(p, y)
        print('  n=%-5d fast=%.10f brute=%.10f %s' % (n, a, b, '✓' if abs(a - b) < 1e-12 else '✗'))

    print('\n=== 边界 ===')
    print('  完美预测      CI=%.4f (期望 1.0)' % ci_harrell(np.arange(10.), np.arange(10.)))
    print('  完全反向      CI=%.4f (期望 0.0)' % ci_harrell(-np.arange(10.), np.arange(10.)))
    print('  全 y 并列     CI=%s (期望 nan)' % ci_harrell(np.arange(10.), np.ones(10)))
    print('  全 p 并列     CI=%s (期望 nan)' % ci_harrell(np.ones(10), np.arange(10.)))
    print('  随机 20 次均值 CI≈%.4f (期望 0.5)' % np.mean(
        [ci_harrell(rng.randn(400), rng.randn(400)) for _ in range(20)]))

    print('\n=== 与 lifelines 对照（若已安装）===')
    try:
        from lifelines.utils import concordance_index
        for n in [100, 800]:
            y = rng.randn(n); p = y + 0.8 * rng.randn(n)
            print('  n=%-5d ours=%.10f lifelines=%.10f' % (
                n, ci_harrell(p, y), concordance_index(y, p)))
    except ImportError:
        print('  lifelines 未安装，跳过')
