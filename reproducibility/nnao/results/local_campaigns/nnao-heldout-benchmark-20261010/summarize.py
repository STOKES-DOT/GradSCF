"""Collect frozen NNAO and independent GPU4PySCF evidence without computing SCF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


BASES = ('minao', 'sto-3g', '3-21g', '6-31g', '6-31g(d)', '6-31g(d,p)',
         'def2-svp', 'cc-pvdz', '6-311g(d,p)', 'def2-tzvp', 'def2-tzvpp',
         'cc-pvtz', 'NNAO-reference', 'NNAO-epoch41')
CUSTOM = BASES[-2:]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [value for value in values if finite(value)]
    return statistics.mean(values) if values else None


def index_results(report):
    result = {}
    for row in report.get('results', []):
        key = row['structure_id'], row['basis']
        if key in result:
            raise ValueError(f'Duplicate result: {key}.')
        result[key] = row
    return result


def timing(rows, field, *, warm=False):
    values = [row.get(field) for row in rows
              if (row.get('index', 0) > 0 if warm else row.get('index', 0) == 0)]
    return mean(values)


def collect(native, gpu4, structures):
    nrows, grows = index_results(native), index_results(gpu4)
    fields = gpu4.get('settings', {})
    for nkey, gkey in (('auxbasis', 'auxbasis'), ('dtype', 'dtype'), ('cart', 'cartesian')):
        if nkey in native and gkey in fields and native[nkey] != fields[gkey]:
            raise ValueError(f'Native and GPU4PySCF {nkey} differ.')
    nhash = native.get('structures_sha256')
    ghash = gpu4.get('provenance', {}).get('structures_sha256')
    if nhash and ghash and nhash != ghash:
        raise ValueError('Native and GPU4PySCF structure file hashes differ.')
    selected = {row['structure_id'] for row in structures}
    if (set(identifier for identifier, _ in nrows) | set(identifier for identifier, _ in grows)) - selected:
        raise ValueError('A result contains a structure outside the selected benchmark.')
    output = []
    for structure in structures:
        identifier = structure['structure_id']
        for basis in BASES:
            n = nrows.get((identifier, basis), {}); g = grows.get((identifier, basis), {})
            custom = basis in CUSTOM
            nstatus = n.get('status', 'pending') if custom else 'not_applicable'
            gstatus = g.get('status', 'pending')
            row = dict(structure_id=identifier, basis=basis,
                natoms=len(structure.get('symbols', [])) or n.get('natoms'),
                native_status=nstatus, gpu4_status=gstatus,
                comparison_status='pending', error=g.get('error', n.get('error', '')),
                nao=g.get('nao', n.get('nao')), primitive_nao=g.get('primitive_nao', n.get('primitive_nao')),
                fock_dimension=g.get('nao', n.get('nao')), fock_elements=None,
                energy_hartree=None, native_energy_hartree=None, native_energy_error_hartree=None,
                above_cc_pvtz_millihartree=None, reference_to_epoch41_gain_millihartree=None,
                native_nn_cold_seconds=timing(n.get('nn_forward_timings', []), 'elapsed_seconds'),
                native_nn_warm_seconds=timing(n.get('nn_forward_timings', []), 'elapsed_seconds', warm=True),
                native_basis_binding_to_cpu_seconds=n.get('basis_binding_and_to_cpu_seconds'),
                native_integral_seconds=n.get('integral_generation_seconds'),
                native_host_device_transfer_seconds=n.get('host_device_transfer_seconds'),
                native_scf_cold_seconds=timing(n.get('scf_timings', []), 'elapsed_seconds'),
                native_scf_warm_seconds=timing(n.get('scf_timings', []), 'elapsed_seconds', warm=True),
                gpu4_setup_seconds=g.get('setup_seconds'),
                gpu4_scf_cold_seconds=timing(g.get('timings', []), 'scf_seconds'),
                gpu4_scf_warm_seconds=timing(g.get('timings', []), 'scf_seconds', warm=True),
                native_scf_cycles=None, gpu4_scf_cycles=None)
            if row['nao'] is not None:
                row['fock_elements'] = row['nao'] ** 2
            if n.get('scf_timings'):
                row['native_scf_cycles'] = n['scf_timings'][-1].get('scf', {}).get('scf_cycles')
            if g.get('timings'):
                row['gpu4_scf_cycles'] = g['timings'][-1].get('scf_cycles')
            if custom and nstatus == 'passed' and finite(n.get('energy_hartree')):
                row['native_energy_hartree'] = n['energy_hartree']
            if gstatus == 'passed' and finite(g.get('energy_hartree')):
                valid = True
                if custom:
                    valid = row['native_energy_hartree'] is not None
                    if valid:
                        error = abs(g['energy_hartree'] - row['native_energy_hartree'])
                        row['native_energy_error_hartree'] = error
                        valid = error <= 1e-7 and n.get('nao') == g.get('nao')
                    if not valid:
                        row['error'] = 'Native/GPU4PySCF same-basis energy or AO-count validation failed.'
                if valid:
                    row.update(comparison_status='passed', energy_hartree=g['energy_hartree'])
                else:
                    row['comparison_status'] = 'failed'
            elif gstatus == 'failed' or (custom and nstatus == 'failed'):
                row['comparison_status'] = 'failed'
            elif gstatus == 'passed':
                row.update(comparison_status='failed', error='Passed result has nonfinite or missing energy.')
            output.append(row)
    by_case = {(row['structure_id'], row['basis']): row for row in output}
    for structure in structures:
        identifier = structure['structure_id']
        anchor = by_case[identifier, 'cc-pvtz']['energy_hartree']
        reference = by_case[identifier, CUSTOM[0]]['energy_hartree']
        trained = by_case[identifier, CUSTOM[1]]['energy_hartree']
        for basis in BASES:
            row = by_case[identifier, basis]
            if anchor is not None and row['energy_hartree'] is not None:
                row['above_cc_pvtz_millihartree'] = 1000 * (row['energy_hartree'] - anchor)
        if reference is not None and trained is not None:
            by_case[identifier, CUSTOM[1]]['reference_to_epoch41_gain_millihartree'] = 1000 * (reference - trained)
    return output


def figures(rows, identifiers, output):
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='nnao-report-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    import numpy as np
    by_case = {(row['structure_id'], row['basis']): row for row in rows}
    gaps = np.array([[by_case[identifier, basis]['above_cc_pvtz_millihartree']
                      if finite(by_case[identifier, basis]['above_cc_pvtz_millihartree']) else np.nan
                      for identifier in identifiers] for basis in BASES])
    limit = max(10., float(np.nanmax(np.abs(gaps)))) if np.isfinite(gaps).any() else 10.
    norm = SymLogNorm(linthresh=10., vmin=-limit, vmax=limit)
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('#e6e6e6')
    fig, axes = plt.subplots(2, 1, figsize=(max(10., len(identifiers) * 1.25), 10.5),
                             gridspec_kw={'height_ratios': [3.5, 1.3]}, layout='constrained')
    heat = axes[0].imshow(np.ma.masked_invalid(gaps), aspect='auto', cmap=cmap, norm=norm)
    axes[0].set_yticks(range(len(BASES)), BASES, fontsize=9)
    axes[0].set_xticks(range(len(identifiers)), identifiers, rotation=35, ha='right', fontsize=8)
    axes[0].set_title('DF-RHF energy relative to the same-geometry cc-pVTZ finite basis')
    for i, basis in enumerate(BASES):
        for j, identifier in enumerate(identifiers):
            value = gaps[i, j]
            if np.isfinite(value):
                label = f'{value:.1f}' if abs(value) < 1000 else f'{value:.0f}'
                color = 'white' if norm(value) < .15 or norm(value) > .85 else 'black'
            else:
                label = 'fail' if by_case[identifier, basis]['comparison_status'] == 'failed' else 'n/a'
                color = '#555555'
            axes[0].text(j, i, label, ha='center', va='center', fontsize=7.5, color=color)
    fig.colorbar(heat, ax=axes[0], label='Energy difference / mHa (signed symmetric-log colors)', shrink=.9)
    gains = [by_case[identifier, CUSTOM[1]]['reference_to_epoch41_gain_millihartree'] for identifier in identifiers]
    valid = [i for i, value in enumerate(gains) if finite(value)]
    axes[1].bar(valid, [gains[i] for i in valid], color='#217c74', width=.6)
    for i, value in enumerate(gains):
        if not finite(value):
            axes[1].text(i, 0., 'n/a', ha='center', va='bottom', color='#555555')
    axes[1].axhline(0., color='#555555', linewidth=.8)
    axes[1].set_xlim(-.6, len(identifiers) - .4)
    axes[1].set_xticks(range(len(identifiers)), identifiers, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Energy reduction / mHa')
    axes[1].set_title('Frozen epoch 41 versus reference contractions; positive means lower energy')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'energy-gap.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(max(12., len(identifiers) * 1.3), 8.3), layout='constrained')
    x = np.arange(len(identifiers)); width = .36
    trained = [by_case[identifier, CUSTOM[1]] for identifier in identifiers]
    def panel(axis, keys, labels, title, backend):
        for offset, key, label in zip((-.5, .5), keys, labels):
            values = [row[key] if row[backend + '_status'] == 'passed' and finite(row[key]) else np.nan
                      for row in trained]
            axis.bar(x + offset * width, values, width, label=label,
                     color='#3776a8' if offset < 0 else '#70b5b3')
        axis.set_title(title, fontsize=10); axis.set_ylabel('Seconds')
        axis.set_xticks(x, identifiers, rotation=40, ha='right', fontsize=7)
        axis.margins(y=.22); axis.legend(frameon=False, fontsize=8)
    panel(axes[0, 0], ('native_nn_cold_seconds', 'native_nn_warm_seconds'),
          ('First call incl. JIT', 'Warm mean'), 'GradSCF: MACE prediction', 'native')
    stages = (('native_basis_binding_to_cpu_seconds', 'Bind + send to CPU', '#80759b'),
              ('native_integral_seconds', 'CPU S/H/DF + whiten/unpack', '#b08451'),
              ('native_host_device_transfer_seconds', 'CPU-to-SCF transfer', '#70b5b3'))
    for offset, (key, label, color) in zip((-1, 0, 1), stages):
        values = [row[key] if row['native_status'] == 'passed' and finite(row[key]) else np.nan
                  for row in trained]
        axes[0, 1].bar(x + offset * .22, values, color=color, width=.22, label=label)
    axes[0, 1].set_title('GradSCF: integral preparation and explicit transfers', fontsize=10)
    axes[0, 1].set_ylabel('Seconds'); axes[0, 1].set_xticks(x, identifiers, rotation=40, ha='right', fontsize=7)
    axes[0, 1].margins(y=.25); axes[0, 1].legend(frameon=False, fontsize=7.5)
    panel(axes[1, 0], ('native_scf_cold_seconds', 'native_scf_warm_seconds'),
          ('First call incl. JIT', 'Warm mean (2 calls)'), 'GradSCF: JAX SCF + validation diagnostics', 'native')
    panel(axes[1, 1], ('gpu4_scf_cold_seconds', 'gpu4_scf_warm_seconds'),
          ('First kernel incl. DF setup', 'Warm kernel (1 call)'), 'GPU4PySCF: fixed-basis SCF kernel', 'gpu4')
    fig.suptitle('Frozen NNAO epoch 41: separate timing scopes, not a backend speedup comparison', fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'timing.{suffix}', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--gpu4', type=Path, required=True)
    parser.add_argument('--structures', type=Path, default=Path(__file__).with_name('test-selection.jsonl'))
    parser.add_argument('--selection-manifest', type=Path,
                        default=Path(__file__).with_name('selection-manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    native = json.loads(args.native.read_text()); gpu4 = json.loads(args.gpu4.read_text())
    structures = [json.loads(line) for line in args.structures.read_text().splitlines() if line.strip()]
    identifiers = [row['structure_id'] for row in structures]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        parser.error('Selected structures must be nonempty with unique IDs.')
    selection_manifest = (json.loads(args.selection_manifest.read_text())
                          if args.selection_manifest.is_file() else {})
    structures_hash = hashlib.sha256(args.structures.read_bytes()).hexdigest()
    isolated = (selection_manifest.get('selected_structures_sha256') == structures_hash
                and selection_manifest.get('structure_ids') == identifiers)
    rows = collect(native, gpu4, structures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gains = [row['reference_to_epoch41_gain_millihartree'] for row in rows if row['basis'] == CUSTOM[1]]
    stats = {basis: dict(valid_energy_structures=sum(row['basis'] == basis and row['comparison_status'] == 'passed'
                for row in rows), anchored_structures=sum(row['basis'] == basis and
                finite(row['above_cc_pvtz_millihartree']) for row in rows),
                mean_above_cc_pvtz_millihartree=mean(row['above_cc_pvtz_millihartree']
                    for row in rows if row['basis'] == basis)) for basis in BASES}
    errors = [row['native_energy_error_hartree'] for row in rows
              if row['comparison_status'] == 'passed' and finite(row['native_energy_error_hartree'])]
    summary = dict(status='complete' if all(row['comparison_status'] == 'passed' for row in rows) else 'partial',
        structure_count=len(identifiers), planned_energy_cases=len(rows),
        checkpoint_epoch=native.get('checkpoint_epoch', 41),
        selection_manifest_matches_structures=isolated,
        selection_manifest_sha256=(hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
                                    if selection_manifest else None),
        passed_energy_cases=sum(row['comparison_status'] == 'passed' for row in rows),
        failed_energy_cases=sum(row['comparison_status'] == 'failed' for row in rows),
        pending_energy_cases=sum(row['comparison_status'] == 'pending' for row in rows),
        paired_gain_structures=sum(finite(value) for value in gains),
        mean_reference_to_epoch41_gain_millihartree=mean(gains),
        max_passed_native_gpu4_energy_error_hartree=max(errors) if errors else None,
        energy_reference='Same-geometry cc-pVTZ finite-basis DF-RHF, not CBS or GMTKN55 reference labels',
        input_hashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
            (('native', args.native), ('gpu4', args.gpu4), ('structures', args.structures))},
        per_basis=stats)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    def display(value, digits=2):
        return f'{value:.{digits}f}' if finite(value) else 'n/a'
    parity_error = summary['max_passed_native_gpu4_energy_error_hartree']
    parity_text = f'{parity_error:.3e} Ha' if finite(parity_error) else 'n/a'
    by_case = {(row['structure_id'], row['basis']): row for row in rows}
    lines = ['# 冻结 NNAO 的独立 HF 测试', '',
        f'**共享 MACE 模型仅训练到第 {summary["checkpoint_epoch"]} 轮；本次冻结权重推理，'
        '没有基组重新拟合、梯度计算或优化器更新。** '
        f'选取 {len(identifiers)} 个结构，与 12 个传统基组及 NNAO-reference／NNAO-epoch41 比较。', '',
        ('当前结构文件与选样清单的哈希及 ID 顺序一致。按选样记录，结构 ID、坐标签名及整体分子式均排除了 '
         '200 个训练结构、50 个验证结构、24 个 pilot 结构和此前苯胺单体系训练。'
         if isolated else '当前输入未与正式选样清单匹配，不能仅根据本报告声称其与训练／验证数据隔离。')
        + '这不保证片段、化学图或反应关联的隔离；例如测试二聚体的单体可能已在训练数据中出现。', '',
        f'有效能量 **{summary["passed_energy_cases"]}/{len(rows)}**；'
        f'{summary["failed_energy_cases"]} 项失败，{summary["pending_energy_cases"]} 项尚无结果。'
        '失败、缺失或后端不一致的计算不计入能量得分。', '',
        f'统一使用 RHF／密度拟合、球谐 AO、float64、辅助基 `{native.get("auxbasis", "def2-universal-jkfit")}`。'
        '几何相同，SCF 均从 hcore 开始；能量阈值 1e-12 Ha、梯度阈值 1e-9、最多 200 次迭代。'
        '必要时用 level shift 稳定化，撤回后重新收敛才接受。', '',
        '| 结构 | NNAO AO / primitive AO / Fock | Reference − cc-pVTZ / mHa | Epoch41 − cc-pVTZ / mHa | Reference → epoch41 降低 / mHa | 有效计算 |',
        '|---|---:|---:|---:|---:|---:|']
    for identifier in identifiers:
        row = by_case[identifier, CUSTOM[1]]
        shape = f'{row["nao"]} / {row["primitive_nao"]} / {row["nao"]}×{row["nao"]}' if row['nao'] else 'n/a'
        count = sum(item['structure_id'] == identifier and item['comparison_status'] == 'passed' for item in rows)
        reference_gap = by_case[identifier, CUSTOM[0]]['above_cc_pvtz_millihartree']
        lines.append(f'| {identifier} | {shape} | {display(reference_gap)} | {display(row["above_cc_pvtz_millihartree"])} | '
                     f'{display(row["reference_to_epoch41_gain_millihartree"])} | {count}/14 |')
    lines += ['', f'相对参考收缩系数，epoch41 的平均能量降低为 '
        f'**{display(summary["mean_reference_to_epoch41_gain_millihartree"])} mHa**，'
        f'仅统计 {summary["paired_gain_structures"]} 个有效配对结构；正值表示降低，负值表示升高。'
        f'通过验收的 native／GPU4PySCF 同基组能量最大差为 '
        f'{parity_text}。', '',
        '参照是同几何的 cc-pVTZ 有限基组 DF-HF 能量，不能作为 CBS 或 GMTKN55 高精度标签。'
        '这批选定结构的 HF 测试不等同完整 GMTKN55 或 WTMAD2 评估。'
        '负的参照能量差仅表示低于该有限基组；能量降低不证明力、BSSE 或基组极限精度。', '',
        '计时分开记录：基组绑定及送到 CPU、native CPU 积分生成、CPU 到 SCF 设备的显式传输各自计时；'
        'CPU 积分时间含 DF 白化／解包，不含后续显式设备传输。MACE 与 JAX SCF 首调用包含 JIT 编译。'
        'Native solve 计时还包含电子能量重构、重叠矩阵最小特征值、commutator 和 raw-Phi 等 JIT 内验收诊断；'
        'GPU4PySCF 首次 kernel 包含 DF 初始化，热态 kernel 重用固定基组 DF。各次 SCF 都从 hcore 独立开始。'
        'GPU4PySCF 诊断在 kernel 计时外执行，因此两个 SCF 计时范围不完全相同。'
        '计时图仅显示 epoch41，所有基组数据在 comparison.csv，未计算后端加速比。', '',
        f'Native 设备：`{native.get("device_kinds", native.get("devices", "unknown"))}`，'
        f'作业 `{native.get("slurm_job_id", "unknown")}`；'
        f'GPU4PySCF 设备：`{gpu4.get("provenance", {}).get("gpu_name", "unknown")}`，'
        f'作业 `{gpu4.get("slurm_job_id", "unknown")}`。源码、输入与选样哈希保留在原始文件及 summary.json。', '',
        '![有限基组能量差](energy-gap.png)', '', '![不同范围的独立计时](timing.png)', '',
        f'原始输入：`{args.native.resolve()}`、`{args.gpu4.resolve()}`。']
    (args.output_dir / 'README.md').write_text('\n'.join(lines) + '\n')
    figures(rows, identifiers, args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != 'per_basis'}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
