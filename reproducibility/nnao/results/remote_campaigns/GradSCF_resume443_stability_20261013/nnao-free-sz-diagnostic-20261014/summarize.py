"""Separate individual variational improvements from reaction-error changes."""
import csv
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/'nnao-relative200-20261014'
FACTOR=627.5094740631

def main():
    result=json.loads((OUT/'summary.json').read_text())
    reactions=json.loads((OUT/'reactions.json').read_text())
    original=json.loads((PARENT/'gpu4/comparison.json').read_text())['results']
    lookup={(r['structure_id'],r['basis']):r['energy_hartree']
            for r in original if r['status']=='passed'}
    structures={r['structure_id']:r for r in result['results']}
    rows=[]
    for reaction in reactions:
        ids={t['structure_id'] for t in reaction['terms']}
        valid=all(structures[i]['status'] in ['complete','partial'] and
                  'best_energy_hartree' in structures[i] for i in ids)
        def relative(energies):
            return math.fsum(t['coefficient']*energies[t['structure_id']]
                             for t in reaction['terms'])*FACTOR+reaction['energy_add_kcal_mol']
        baseline=relative({i:lookup[(i,'NNAO-200-best')] for i in ids})
        anchor=relative({i:lookup[(i,'cc-pvtz')] for i in ids})
        free=relative({i:structures[i]['best_energy_hartree'] for i in ids}) if valid else None
        row=dict(reaction_id=reaction['reaction_id'],category=reaction['category'],
            status='available' if valid else 'missing',mace_kcal_mol=baseline,
            finite_budget_free_sz_kcal_mol=free,hf_ccpvtz_kcal_mol=anchor,
            gmtkn55_kcal_mol=reaction['reference_energy_kcal_mol'],
            mace_error_vs_hf_kcal_mol=baseline-anchor,
            free_error_vs_hf_kcal_mol=None if free is None else free-anchor,
            mace_error_vs_gmtkn55_kcal_mol=baseline-reaction['reference_energy_kcal_mol'],
            free_error_vs_gmtkn55_kcal_mol=None if free is None else free-reaction['reference_energy_kcal_mol'])
        if valid:
            summed_gain=math.fsum(t['coefficient']*structures[t['structure_id']]['mace_to_best_millihartree']/1000
                                  for t in reaction['terms'])*FACTOR
            assert abs((baseline-free)-summed_gain)<1e-5,'Stoichiometric improvement identity failed'
        rows.append(row)
    with (OUT/'relative-energies.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_rows=[]
    for record in result['results']:
        atomic_rows.append({k:record.get(k) for k in ['structure_id','status','mace_energy_hartree',
            'best_energy_hartree','best_candidate','mace_to_best_millihartree','completed_starts',
            'converged_starts','endpoint_spread_millihartree','converged_spread_millihartree']})
    with (OUT/'per-structure.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(atomic_rows[0]));writer.writeheader();writer.writerows(atomic_rows)
    lines=['# 固定指数自由 SZ 收缩诊断','',
        '固定 epoch 629 的指数、SZ AO 结构及内层；逐结构优化完整原始收缩系数，不更新 MACE。',
        '每结构四起点、每起点最多 500 步。以下是有限预算内保留原始 MACE 解的最佳值，不是全局收缩极限。','',
        '| 条目 | MACE | 自由 SZ 最佳 | HF/cc-pVTZ | GMTKN55 |',
        '|---|---:|---:|---:|---:|']
    fmt=lambda x:'缺失' if x is None else f'{x:.6f}'
    for row in rows:
        lines.append('| '+row['reaction_id']+' | '+' | '.join(fmt(row[k]) for k in
            ['mace_kcal_mol','finite_budget_free_sz_kcal_mol','hf_ccpvtz_kcal_mol','gmtkn55_kcal_mol'])+' |')
    lines += ['', '单位 kcal/mol；S22 沿用正结合能符号，未做 counterpoise 校正。','',
        '## 优化可靠性','',
        '| 结构 | 相对 MACE 降低 (mEh) | 收敛起点/4 | 全部终点能量跨度 (mEh) | 最佳来源 |',
        '|---|---:|---:|---:|---|']
    for r in atomic_rows:
        lines.append(f"| {r['structure_id']} | {fmt(r['mace_to_best_millihartree'])} | {r['converged_starts']} | {fmt(r['endpoint_spread_millihartree'])} | {r['best_candidate']} |")
    lines += ['', '达到步数上限或更新被拒绝不等于收敛；起点差异显著时优先处理优化可靠性。',
        '总能量下降不保证相对能量误差下降。单体系自由收缩比共享 MACE 具有更大的输出自由度；本实验不能独自区分表达能力、训练不足和泛化。',
        '具体收敛状态、梯度残差、有限差分检查、失败和每次接受步骤见 summary.json 及 checkpoints/。']
    verification=OUT/'gpu4-verification/comparison.json'
    if verification.exists():
        v=json.loads(verification.read_text())
        errors=[r['native_energy_error_hartree'] for r in v['results'] if r['status']=='passed']
        from collections import Counter
        counts=Counter(s['status'] for r in result['results'] for s in r['starts'])
        checks=[c for r in result['results'] for c in r.get('gradient_checks',[])]
        lines += ['', '## 数值核验与状态', '',
            f"- 独立 GPU4PySCF 复核通过 {v['passed']}/7，失败 {v['failed']}；最大能量差 {max(errors):.3e} Eh。",
            f"- 方向有限差分 {sum(c['passed'] for c in checks)}/{len(checks)} 通过，最大导数差 {max(c['absolute_error'] for c in checks):.3e}。",
            f'- 起点停止状态：{dict(counts)}。',
            f"- 优化阶段用时 {result['elapsed_seconds']:.1f} 秒；设备 {result['device']}，CPU {result['cpu_count']}。", '',
            '## 当前判断', '',
            '三个已选诊断条目均能在不增加 AO 数量、不改变指数的情况下显著改善相对能量。当前 MACE 收缩输出尚未充分利用既有基组空间。',
            '该结果不能单独区分网络表达、训练不足与泛化，也不是对全部 18 项测试的结论。',
            'H2 的四个起点均未达到梯度阈值，终点差约 2.06 mEh；HCN 过渡态也未达到梯度阈值。应先改善这些优化结果的可靠性，再把自由 SZ 与同池 DZ、指数优化作结构极限比较。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(relative_energies=rows,structures=atomic_rows),indent=2))

if __name__=='__main__':
    main()
