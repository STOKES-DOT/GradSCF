"""Assemble the converged A/B comparison without modifying production exponents."""
import json
from pathlib import Path
import shutil

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.model.nnao import prepare_direct_basis
from gradscf.model.nnao.initialization import with_pool_parameters

output = Path('artifacts/pool-free-convergence-20261012')
parent = Path('artifacts/pool-calibration-refined-20261012')
summary = json.loads((output/'summary.json').read_text())
metadata = json.loads((parent/'metadata.json').read_text())
by_candidate = {}
for row in summary['comparisons']:
    by_candidate.setdefault(row['candidate'], {})[row['structure_id']] = row
weights = {x['structure_id']: x['weight'] for x in metadata['references']['structures']}
records = []
for identifier, base in by_candidate['baseline'].items():
    candidate = by_candidate['calibrated_candidate'][identifier]
    records.append(dict(structure_id=identifier, pool_nao=base['pool_nao'],
        compact_nao=base['contracted_nao'],
        baseline_gap_millihartree=1000*base['delta_contraction_hartree'],
        candidate_gap_millihartree=1000*candidate['delta_contraction_hartree'],
        free_energy_gain_millihartree=1000*(base['free_energy_hartree']-candidate['free_energy_hartree']),
        both_free_optimizations_converged=base['optimizer_converged'] and candidate['optimizer_converged']))
average = lambda candidate, key: sum(weights[k]*v[key] for k, v in by_candidate[candidate].items())/sum(weights.values())
analysis = dict(records=records, candidate_is_provisional=True, production_exponents_changed=False,
    mean_pool_gain_millihartree=1000*(average('baseline', 'pool_energy_hartree')-average('calibrated_candidate', 'pool_energy_hartree')),
    mean_free_gain_millihartree=1000*(average('baseline', 'free_energy_hartree')-average('calibrated_candidate', 'free_energy_hartree')),
    baseline_mean_gap_millihartree=1000*average('baseline', 'delta_contraction_hartree'),
    candidate_mean_gap_millihartree=1000*average('calibrated_candidate', 'delta_contraction_hartree'))
with np.load(parent/'checkpoints/pool-calibration.npz') as saved:
    point, gradient = saved['parameters'], saved['gradient']
    analysis['pool_projected_gradient_norm'] = float(np.linalg.norm(point-np.clip(point-gradient, -.15, .15)))
(output/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
lines = ['# NNAO 初始 primitive 池：首批 A/B 对照', '',
    '2026-10-12；CPU float64，真实 native DF 积分与 implicit RHF。',
    '固定内层基组参数，全部电子参与 SCF。spherical szp663_direct/core6；def2-universal-jkfit。', '',
    '| 分子 | 池 AO → compact AO | 原始池收缩差 / mEh | 候选池收缩差 / mEh | 候选自由收缩能量降低 / mEh |',
    '| --- | ---: | ---: | ---: | ---: |']
for row in records:
    lines.append(f"| {row['structure_id']} | {row['pool_nao']} → {row['compact_nao']} | "
        f"{row['baseline_gap_millihartree']:.5f} | {row['candidate_gap_millihartree']:.5f} | {row['free_energy_gain_millihartree']:.5f} |")
lines += ['', f"平均未收缩能量降低 {analysis['mean_pool_gain_millihartree']:.5f} mEh；平均自由收缩能量降低 {analysis['mean_free_gain_millihartree']:.5f} mEh。",
    f"平均收缩差从 {analysis['baseline_mean_gap_millihartree']:.5f} 降到 {analysis['candidate_mean_gap_millihartree']:.5f} mEh。", '',
    '## 优化状态与解释', '',
    '每个候选／分子使用三个收缩初值。表中十二个最低能量结果均满足原始系数梯度范数 < 1e-5，SCF 均收敛并通过内部稳定性检查。',
    'NH3/H2O/CO 候选从第 300 步保存的参数、梯度和 Adam 状态续算；恢复前重新核对同一点的能量与梯度，分别在最佳起点的 559/505/538 个接受更新后达到阈值。',
    '这些是多起点局部优化结果，不是已证明的全局收缩极限。',
    '池指数标定接受了 16 个更新；第 17 个 Adam 提案及缩短步未能通过能量下降检查，因此保留最后有效参数。',
    f"池的投影梯度范数仍为 {analysis['pool_projected_gradient_norm']:.6g}；不能称为完全优化的指数池。", '',
    '候选在这六个分子的自由收缩能量上均优于原始池，但 CH4 的收缩差略大，说明未收缩能量与可压缩性必须分开检查。',
    'CO 的收缩差仍最大；后续应检查 primitive 分布与目标单径向函数结构的匹配，不能归因于尚未训练的 MACE。',
    '该候选保留供下一轮验证，未替换正式初始化指数；没有训练 MACE，也没有测试偶极、极化率、BSSE 或泛化。', '',
    '## 证据', '',
    f'- 原始实验：`{parent}`，包含完整参考几何、参数、来源哈希、全部起点和逐阶段计时／RSS。',
    '- 本目录：恢复后的检查点、`continuation.json`、`summary.json`、`analysis.json`。',
    '- `continue_free.py` 位于原始实验目录；本文件 `analyze.py` 重建本比较表。',
    '- `candidate-exponents.json` 保存候选的绝对指数值，标记为 provisional。', '']
(output/'COMPARISON.md').write_text('\n'.join(lines))
keys = tuple(map(tuple, metadata['channel_keys']))
pool = json.loads((parent/'calibrated_candidate-parameters.json').read_text())
channels = {}
for row in metadata['references']['structures']:
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                  cart=False, basis_family='szp663_direct', core_primitives=6)
    moved = with_pool_parameters(layout, np.asarray(pool['values']), keys)
    for atom, l, role, slot, old, new in zip(layout.shell_atoms, layout.topology.angular_momenta,
            layout.roles, layout.slots, layout.parameters.exponents, moved.parameters.exponents):
        if slot < 0:
            np.testing.assert_array_equal(old, new)
            continue
        key = (layout.symbols[atom], l, role)
        channels[key] = dict(element=key[0], l=l, role=role,
            reference_exponents_bohr_minus2=np.asarray(old).tolist(),
            candidate_exponents_bohr_minus2=np.asarray(new).tolist())
(output/'candidate-exponents.json').write_text(json.dumps(dict(provisional=True,
    production_exponents_changed=False, channels=list(channels.values()),
    parameters=pool, fixed_core='unchanged six-primitive core; all-electron SCF'), indent=2)+'\n')
for name in ('metadata.json', 'basis-inventory.json', 'exponent-gradient-check.json'):
    shutil.copy2(parent/name, output/name)
print(json.dumps(analysis, indent=2))
