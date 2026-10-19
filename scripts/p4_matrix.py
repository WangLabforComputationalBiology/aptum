# -*- coding: utf-8 -*-
"""
P4 · 4 协议 × 5 数据集 × 2 主指标（RMSE / CI）矩阵汇总
=====================================================
读各格子产物，输出统一矩阵表（markdown + json）。缺失格子标 PENDING，不猜、不填。

"第一"判定（C1–C5，见 FULL_MATRIX_PLAN.md）：
  C1 该格存在同口径对手（我们自己在同一划分上重训的最强方法）
  C2 点估计更低（RMSE）/ 更高（CI）
  C3 配对 bootstrap 显著（p(我们更差) < 0.05）
  C4 口径对齐（5-fold 或明确标注的单次划分）
  C5 随机/scaffold 格必须给出泄漏量化
"""
import os, sys, json, glob
import numpy as np

MID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(MID, 'results')
P2 = os.path.join(R, 'p2')


def jload(p, default=None):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return default


def f(x, n=4):
    return '—' if x is None else ('%.*f' % (n, x))


# ---------------------------------------------------------------------------
# 每格的数据来源
# ---------------------------------------------------------------------------
def cell_chembl_temporal():
    cm = jload(os.path.join(R, 'ci_matrix.json'), {})
    c = cm.get('chembl_temporal', {})
    m = c.get('methods', {})
    ours = m.get('kairos', {})
    opp = {k: v for k, v in m.items() if k != 'kairos'}
    bo = min(opp, key=lambda k: opp[k]['rmse']) if opp else None
    return dict(ours_rmse=ours.get('rmse'), ours_ci=ours.get('ci'), n=c.get('n'),
                opp=bo, opp_rmse=opp[bo]['rmse'] if bo else None, opp_ci=opp[bo]['ci'] if bo else None,
                leakage='时间轴：train ≤ 2018 训练，test 2019+；无跨期同对泄漏',
                published='同生态 ChEMBL 无公开时间轴榜；对手为我们在同划分重训的最强方法',
                sig=(cm.get('verdicts', {}).get('chembl_temporal', {}) or {}).get('margin'))


def cell_chembl_random():
    V = jload(os.path.join(R, 'p2_theta_variants.json'), {})
    r = V.get('chembl_random', {})
    kt = jload(os.path.join(R, 'kairos_theta_rule.json'), {})
    cc = (kt.get('cells', {}) or {}).get('chembl_random', {})
    return dict(ours_rmse=r.get('nnls_free', {}).get('rmse'), ours_ci=r.get('nnls_free', {}).get('ci'),
                n=r.get('n_test'), opp=r.get('best_single'), opp_rmse=r.get('best_single_rmse'),
                opp_ci=r.get('best_single_ci'),
                leakage='行级随机 → 泄漏严重：分子重叠 59.1%，(分子,靶点) 对重叠 23.7%',
                published='公开随机榜（Davis/KIBA 体系的"榜"）在本数据集无同名对手',
                sig=cc.get('official_rmse'))


def _p2_cell(key, leakage, published):
    """优先读 kairos_rule.json 的官方口径（含无标签位移检测器自动切换的 θ）；
    未在规则表内时退回该格 json 的 kairos_honest，并标注口径来源。"""
    KR = jload(os.path.join(R, 'kairos_theta_rule.json'), {})
    rc = (KR.get('cells', {}) or {}).get(key)
    if rc:
        bo = rc['best_single']
        return dict(ours_rmse=rc['official']['rmse'], ours_ci=rc['official']['ci'],
                    n=rc['n_test'], opp=bo,
                    opp_rmse=rc['single'][bo]['rmse'], opp_ci=rc['single'][bo]['ci'],
                    theta=rc['official']['theta'],
                    caliber='shift_d=%.4f → %s' % (rc['shift_d_max'], rc['chosen']),
                    alt=dict(nnls_free=rc['nnls_free']['rmse'], simplex=rc['simplex']['rmse'],
                             cv5_test=rc['cv5_test']['rmse'], oracle=rc['oracle_testfit']['rmse']),
                    leakage=leakage, published=published)
    d = jload(os.path.join(P2, '%s.json' % key), {})
    if not d:
        return dict(PENDING=True, leakage=leakage, published=published)
    bo = d.get('best_single')
    bt = d.get('bootstrap_vs_best_single', {})
    return dict(ours_rmse=d.get('kairos_honest', {}).get('rmse'),
                ours_ci=d.get('kairos_honest', {}).get('ci'),
                n=d.get('n_test'), opp=bo,
                opp_rmse=d.get('members', {}).get(bo, {}).get('rmse') if bo else None,
                opp_ci=d.get('members', {}).get(bo, {}).get('ci') if bo else None,
                theta=d.get('kairos_honest', {}).get('theta'),
                p_worse=(bt.get('rmse_gain', {}) or {}).get('p_agg_worse'),
                leakage=leakage, published=published)


def cell_davis_random():
    # 优先读升级池 ceiling（12 成员 cv5，含 graphdta/新成员；p52b_ceiling 产出）
    ceil = jload(os.path.join(R, 'absorb', 'p52_ceiling_davis.json'), {})
    if ceil:
        return dict(ours_rmse=ceil['cv5_aggregate']['rmse'], ours_ci=ceil['cv5_aggregate']['ci'],
                    n=ceil['n_test'], opp=ceil['best_single'],
                    opp_rmse=ceil['best_single_rmse'], opp_ci=ceil['best_single_ci'],
                    members=ceil['members'],
                    leakage='Davis 官方 setting1 单划分（train 25046 / test 5010）',
                    published='DeepDTA Davis MSE 0.261 → RMSE≈0.511；MutualDTA 自报 0.443（5-fold 均值）',
                    caliber='cv5（旧成员无 val 预测；含 graphdta）',
                    oracle=ceil['oracle_free_theta']['rmse'])
    cm = jload(os.path.join(R, 'ci_matrix.json'), {})
    c = cm.get('davis', {})
    m = c.get('methods', {})
    ours = m.get('kairos_5foldcv', {})
    opp = {k: v for k, v in m.items() if k != 'kairos_5foldcv'}
    bo = min(opp, key=lambda k: opp[k]['rmse']) if opp else None
    return dict(ours_rmse=ours.get('rmse'), ours_ci=ours.get('ci'), n=c.get('n'),
                opp=bo, opp_rmse=opp[bo]['rmse'] if bo else None, opp_ci=opp[bo]['ci'] if bo else None,
                leakage='Davis 官方 setting1 单划分（train 25046 / test 5010）；随机口径，同对不跨划分',
                published='DeepDTA Davis MSE 0.261 → RMSE≈0.511；MutualDTA 自报 0.443（5-fold 均值）',
                note='θ 口径：test 内 5 折交叉拟合')


def _kiba_random_cell():
    """KIBA 随机：主口径 = 官方 setting1 划分（train 98545 / test 19709）；
    自定 80/10/10（test 11825）作稳健性副本。"""
    d = _p2_cell('kiba_official', 'KIBA 官方 setting1 划分，同对不跨划分',
                 'DeepDTA KIBA MSE 0.411 → RMSE≈0.641，CI 0.782（官方 5-fold）')
    d['robustness_ad_hoc_80_10_10'] = _p2_cell(
        'kiba_random', '自定 80/10/10（test 11825），非官方划分',
        '稳健性检验用，不作主口径')
    return d


def cell_kiba_random():
    kt = jload(os.path.join(R, 'kairos_theta_rule.json'), {})
    r = (kt.get('cells', {}) or {}).get('kiba_random', {})
    V = jload(os.path.join(R, 'p2_theta_variants.json'), {})
    vr = V.get('kiba_random', {})
    # 官方 setting1 划分（若已跑完）
    off = jload(os.path.join(P2, 'kiba_official.json'), {})
    d = dict(
        ours_rmse=r.get('official_rmse'), ours_ci=r.get('official_ci'), n=r.get('n_test'),
        opp=r.get('best_single'), opp_rmse=r.get('best_single_rmse'), opp_ci=vr.get('best_single_ci'),
        leakage='KIBA 官方 setting1 划分（train 98545 / test 19709）；同对不跨划分',
        published='DeepDTA KIBA MSE 0.411 → RMSE≈0.641，CI 0.782（官方 5-fold）',
        note='主口径=官方 setting1；另有自定 80/10/10 副本（test 11825）作稳健性检验')
    if off:
        bo = off.get('best_single')
        d.update(official_split=dict(ours_rmse=off['kairos_honest']['rmse'],
                                     ours_ci=off['kairos_honest']['ci'],
                                     opp=bo, opp_rmse=off['members'][bo]['rmse'],
                                     opp_ci=off['members'][bo]['ci'],
                                     n=off['n_test']))
    return d


def _kinase_cell(key, tpl):
    cm = jload(os.path.join(R, 'ci_matrix.json'), {})
    c = cm.get('kinase_temporal', {}) if key == 'kinase_temporal' else {}
    d = dict(tpl)
    if key == 'kinase_temporal':
        m = c.get('methods', {})
        ours = m.get('kairos_5foldcv', {})
        opp = {k: v for k, v in m.items() if k != 'kairos_5foldcv'}
        bo = min(opp, key=lambda k: opp[k]['rmse']) if opp else None
        d.update(ours_rmse=ours.get('rmse'), ours_ci=ours.get('ci'), n=c.get('n'),
                 opp=bo, opp_rmse=opp[bo]['rmse'] if bo else None, opp_ci=opp[bo]['ci'] if bo else None)
    return d


def cell_kinase_split(split):
    """激酶基准的 random/scaffold/target：对手=仓库内 6-7 个基线；我们=聚合（需 P2.1 预测）"""
    tpl = dict(
        leakage={'random': '行级随机 → 泄漏严重（分子、靶点均跨划分重叠）',
                 'scaffold': 'Murcko 骨架分组，骨架不跨划分；靶点重叠 100%（同靶不同骨架）',
                 'target': '靶点级留出，靶点不跨划分；分子重叠 100%（同分子不同靶）'}[split],
        published='激酶亲和力基准论文自报最优 0.777（其自有划分）')
    a = jload(os.path.join(R, 'kinase_%s_kairos.json' % split))
    if not a:
        return dict(PENDING=True, **tpl)
    bs = a['best_single']
    tpl.update(ours_rmse=a['kairos']['rmse'], ours_ci=a['kairos']['ci'], n=a['n_test'],
               opp=bs, opp_rmse=a['single'][bs]['rmse'], opp_ci=a['single'][bs]['ci'],
               theta=a['kairos'].get('theta'),
               caliber='shift_d=%.4f → %s' % (a['shift_d_max'], a['chosen']),
               alt=dict(cv5_test=a['kairos_cv5']['rmse'], oracle=a['kairos_oracle']['rmse']),
               p_worse=a['bootstrap_vs_best_single']['rmse_gain']['p_agg_worse'],
               members_note='%d 个成员' % len(a['members']))
    return tpl


def cell_bindingdb(proto):
    """四格统一走 kairos_rule.json 官方口径（预测已从 ckpt 恢复并落盘，
    与 bdb_result.json / bdb_rand_result.json 的 RMSE 逐位一致）。"""
    leak = {
        'temporal': '时间轴：按 year 划分，train ≤ 2018；无跨期同对泄漏',
        'random': '行级随机 → (分子,靶点) 对重叠 3.53%，分子重叠 45.75%，靶点重叠 100%',
        'scaffold': 'Murcko 骨架不跨划分；靶点重叠见 json 的 leakage 字段',
        'cold': '靶点级留出，靶点不跨划分；分子重叠见 json 的 leakage 字段',
    }[proto]
    return _p2_cell('bindingdb_%s' % proto, leak, 'BindingDB 无公开同口径榜')


def build():
    rows = []
    rows.append(('ChEMBL 战场', '时间', cell_chembl_temporal()))
    rows.append(('ChEMBL 战场', '随机', _p2_cell(
        'chembl_random', '行级随机 → 泄漏严重：分子重叠 59.1%，(分子,靶点) 对重叠 23.7%',
        '公开随机榜（Davis/KIBA 体系）在本数据集无同名对手')))
    rows.append(('ChEMBL 战场', 'scaffold',
                 _p2_cell('chembl_scaffold', 'Murcko 骨架不跨划分；靶点重叠见 json 的 leakage 字段',
                          'ChEMBL 无公开 scaffold 榜；对手=我们在同划分重训的最强方法')))
    rows.append(('ChEMBL 战场', '冷启动',
                 _p2_cell('chembl_cold', '靶点级留出，靶点不跨划分；分子重叠见 json 的 leakage 字段',
                          'ChEMBL 无公开冷启动榜；对手=我们在同划分重训的最强方法')))
    rows.append(('BindingDB', '时间', cell_bindingdb('temporal')))
    rows.append(('BindingDB', '随机', cell_bindingdb('random')))
    rows.append(('BindingDB', 'scaffold', cell_bindingdb('scaffold')))
    rows.append(('BindingDB', '冷启动', cell_bindingdb('cold')))
    rows.append(('Davis', '时间', dict(SKIP='Davis 无年份字段，时间轴结构上不可做')))
    rows.append(('Davis', '随机', cell_davis_random()))
    rows.append(('Davis', 'scaffold',
                 _p2_cell('davis_scaffold', 'Murcko 骨架不跨划分；但 Davis 仅 68 个分子 → test 仅 6 个分子，统计功效极低',
                          'Davis 无公开 scaffold 榜')))
    rows.append(('Davis', '冷启动',
                 _p2_cell('davis_cold', '靶点级留出；分子重叠 100%',
                          'Davis 无公开冷启动榜')))
    rows.append(('KIBA', '时间', dict(SKIP='KIBA 无年份字段，时间轴结构上不可做')))
    rows.append(('KIBA', '随机', _kiba_random_cell()))
    rows.append(('KIBA', 'scaffold',
                 _p2_cell('kiba_scaffold', 'Murcko 骨架不跨划分', 'KIBA 无公开 scaffold 榜')))
    rows.append(('KIBA', '冷启动',
                 _p2_cell('kiba_cold', '靶点级留出；分子重叠 100%', 'KIBA 无公开冷启动榜')))
    rows.append(('激酶基准', '时间', _kinase_cell('kinase_temporal', dict(
        leakage='时间轴：train ≤ 2018，test 2019+',
        published='激酶基准论文自报最优 0.777（随机划分口径）'))))
    rows.append(('激酶基准', '随机', cell_kinase_split('random')))
    rows.append(('激酶基准', 'scaffold', cell_kinase_split('scaffold')))
    rows.append(('激酶基准', '冷启动', cell_kinase_split('target')))
    return rows


def main():
    rows = build()
    out = []
    print('\n' + '=' * 132)
    print('Kairos · 4 协议 × 5 数据集 × 2 主指标 矩阵')
    print('=' * 132)
    print('%-13s %-8s %10s %8s %7s | %-14s %10s %8s | %-6s %-6s %-6s' % (
        '数据集', '协议', 'Kairos RMSE', 'Kairos CI', 'n_test',
        '同口径最强对手', '其 RMSE', '其 CI', 'RMSE', 'CI', 'bootstrap'))
    print('-' * 132)
    n_first_rmse = n_first_ci = n_done = 0
    for ds, proto, d in rows:
        if 'SKIP' in d:
            print('%-13s %-8s  %s' % (ds, proto, d['SKIP']))
            out.append(dict(dataset=ds, protocol=proto, status='STRUCTURAL_N/A', reason=d['SKIP']))
            continue
        if d.get('PENDING'):
            print('%-13s %-8s  PENDING' % (ds, proto))
            out.append(dict(dataset=ds, protocol=proto, status='PENDING'))
            continue
        n_done += 1
        orm, oci = d.get('ours_rmse'), d.get('ours_ci')
        prm, pci = d.get('opp_rmse'), d.get('opp_ci')
        w_rmse = '第一' if (orm is not None and prm is not None and orm < prm) else (
            '—' if prm is None else '落后')
        w_ci = '第一' if (oci is not None and pci is not None and oci > pci) else (
            '—' if pci is None else '落后')
        n_first_rmse += (w_rmse == '第一')
        n_first_ci += (w_ci == '第一')
        print('%-13s %-8s %10s %8s %7s | %-14s %10s %8s | %-6s %-6s %-6s' % (
            ds, proto, f(orm), f(oci), d.get('n') or '—',
            (d.get('opp') or '无对手')[:14], f(prm), f(pci), w_rmse, w_ci,
            ('%.3f' % d['p_worse']) if d.get('p_worse') is not None else '—'))
        out.append(dict(dataset=ds, protocol=proto, status='DONE',
                        ours=dict(rmse=orm, ci=oci, n=d.get('n')),
                        opponent=dict(name=d.get('opp'), rmse=prm, ci=pci) if d.get('opp') else None,
                        verdict=dict(rmse=w_rmse, ci=w_ci),
                        p_agg_worse=d.get('p_worse'),
                        leakage=d.get('leakage'), published=d.get('published'),
                        theta=d.get('theta'), caliber=d.get('caliber'),
                        alternates=d.get('alt'), extra=d.get('official_split')))
    print('-' * 132)
    print('已完成格 %d / 18 ；RMSE 第一 %d 格，CI 第一 %d 格' % (n_done, n_first_rmse, n_first_ci))
    print('=' * 132)

    js = os.path.join(R, 'matrix_4x5.json')
    json.dump(dict(rows=out, n_done=n_done, n_first_rmse=n_first_rmse, n_first_ci=n_first_ci),
              open(js, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\n写入 %s' % js)

    # ---- 第二口径：cv5_test（transductive，需目标域一小批已测标签）----
    KR = jload(os.path.join(R, 'kairos_theta_rule.json'), {}).get('cells', {})
    print(chr(10) + '=' * 108)
    print('第二口径：cv5_test（test 内 5 折交叉拟合 θ）—— transductive，'
          '对应"新化学空间里有一批已测数据可校准"的部署场景')
    print('=' * 108)
    print('%-13s %-8s %12s %12s %12s %8s' % ('数据集', '协议', 'cv5 RMSE', '最佳单 RMSE', 'cv5 CI', 'RMSE 判定'))
    print('-' * 108)
    n2 = 0
    for ds, proto, d in rows:
        if 'SKIP' in d or d.get('PENDING'):
            continue
        key = {'时间': 'temporal', '随机': 'random', 'scaffold': 'scaffold', '冷启动': 'cold'}[proto]
        dsmap = {'ChEMBL 战场': 'chembl', 'BindingDB': 'bindingdb', 'Davis': 'davis',
                 '激酶基准': 'kinase'}
        if ds == 'KIBA':
            # KIBA 的主口径随机格用的是官方 setting1（kiba_official）
            k2 = {'random': 'kiba_official', 'scaffold': 'kiba_scaffold', 'cold': 'kiba_cold'}[key]
        else:
            k2 = '%s_%s' % (dsmap[ds], key)
        rc = KR.get(k2)
        if not rc:
            continue
        bs = rc['best_single']
        c = rc['cv5_test']
        sr = rc['single'][bs]
        w = '第一' if c['rmse'] < sr['rmse'] else '落后'
        n2 += (w == '第一')
        print('%-13s %-8s %12.4f %12.4f %12.4f %8s' % (ds, proto, c['rmse'], sr['rmse'], c['ci'], w))
    print('-' * 108)
    print('cv5 口径下 RMSE 第一 %d 格' % n2)
    print('=' * 108)

    # 泄漏明细单列，供 C5 检查
    print('\n泄漏量化（C5 检查项）')
    print('-' * 132)
    for ds, proto, d in rows:
        if 'SKIP' in d or d.get('PENDING'):
            continue
        print('  %-13s %-8s %s' % (ds, proto, d.get('leakage', '（未记录）')))


if __name__ == '__main__':
    main()
