"""Compare the frozen 24-structure best975 model with the verified 10-case baseline.

This collector computes no electronic structure. The job-95106 baseline is
byte-pinned; only the new pilot-best975 native/GPU4PySCF rows are measured anew.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


DIRECTORY = Path(__file__).resolve().parent
STRUCTURES_HASH = '60c3a44e19214decdbc07701c27460b8098e38dc5184e4f44e0e8e14ceba8987'
SCIENCE_HASH = '87f14784cbc038349f65f311c4b664809925932abd2f0cccea834c5400102b30'
PILOT_CHECKPOINT = 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de'
STAGE2_CHECKPOINT = '7e7f66fd900fe50cf9925f6d875fd566a953eb5a67600d67c53c518254084df2'
BASELINE_HASHES = dict(native='b6e236e2b5766dc760d6a3b322b00880c7058d77f4c7c1b417b393a1cba09ce6',
                       gpu4='aaf65b4ebd2900a316b38deba0719d3cea4cbf2b1ffc147d85965901777b7fb7')
REFERENCE = 'NNAO-reference'; STAGE2 = 'NNAO-epoch41'; PILOT = 'NNAO-pilot-best975'
spec = importlib.util.spec_from_file_location('shared_evidence_collector', DIRECTORY / 'summarize.py')
shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
BASES = (*shared.BASES, PILOT)
# This private helper instance validates the initial/reference and newly measured
# model. The immutable stage-two rows are separately checked below, then labeled.
shared.BASES = BASES; shared.CUSTOM = (REFERENCE, PILOT)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reports(bn, bg, native, gpu4):
    """Reject different experiments before calculating any cross-model scores."""
    require(bn.get('status') == 'complete' and bn.get('failed') == 0
            and bg.get('status') == 'passed' and bg.get('failed') == 0,
            'The job-95106 baseline must be completely passed.')
    require(bn.get('checkpoint_epoch') == 41 and bn.get('checkpoint_sha256') == STAGE2_CHECKPOINT,
            'The baseline must be the known stage-two epoch41 checkpoint.')
    require(native.get('checkpoint_profile') == 'pilot-best975'
            and native.get('checkpoint_epoch') == 975
            and native.get('model_training_structure_count') == 24
            and native.get('checkpoint_sha256') == PILOT_CHECKPOINT,
            'New results must identify the original 24-structure pilot best975 model.')
    for report in (bn, native):
        require(report.get('structures_sha256') == STRUCTURES_HASH, 'Native structure source hash differs.')
        require(report.get('provenance', {}).get('repaired_scientific_source_sha256') == SCIENCE_HASH,
                'Native scientific source differs from the verified repaired source.')
        require(report.get('method') == 'RHF' and report.get('dtype') == 'float64'
                and report.get('cart') is False and report.get('auxbasis') == 'def2-universal-jkfit',
                'Native method, auxiliary basis, dtype or angular representation differs.')
        require(report.get('performed_gradients') is False and report.get('performed_optimizer_updates') is False
                and report.get('requested_four_center_eri') is False,
                'This comparison requires frozen forward-only DF inference.')
    expected_scf = dict(max_cycle=200, conv_tol=1e-12, conv_tol_density=1e-10,
                        conv_tol_grad=1e-9, scf_rescue_level_shift=.2, initial_guess='hcore')
    require(bn.get('scf_config') == native.get('scf_config') == expected_scf,
            'Native SCF configuration differs from the baseline.')
    for field in ('model_config', 'parameter_count', 'basis_family', 'core_primitives',
                  'seed', 'native_library_sha256', 'device_kinds', 'versions'):
        require(native.get(field) == bn.get(field), f'Native {field} differs from the baseline.')
    require(bn.get('device_kinds') == ['Tesla V100-SXM2-32GB'], 'The baseline hardware is not V100 32GB.')
    for report in (bg, gpu4):
        require(report.get('provenance', {}).get('structures_sha256') == STRUCTURES_HASH,
                'GPU4PySCF structure source hash differs.')
        require(report.get('performed_gradients') is False and report.get('requested_four_center_eri') is False,
                'GPU4PySCF evidence must use forward-only DF.')
    require(gpu4.get('settings') == bg.get('settings'), 'GPU4PySCF SCF settings differ from the baseline.')
    for field in ('gpu_name', 'versions', 'imported_sources'):
        require(gpu4.get('provenance', {}).get(field) == bg.get('provenance', {}).get(field),
                f'GPU4PySCF {field} differs from the baseline.')
    nbase = shared.index_results(bn); gbase = shared.index_results(bg)
    identifiers = {identifier for identifier, _ in nbase}
    require(len(identifiers) == 10 and len(nbase) == 20 and len(gbase) == 140,
            'The baseline must have 10 structures, 20 native and 140 GPU4 cases.')
    for identifier in identifiers:
        for basis in BASES[:-1]:
            row = gbase.get((identifier, basis), {})
            require(row.get('status') == 'passed' and shared.finite(row.get('energy_hartree')),
                    f'Invalid baseline GPU4 case {identifier}/{basis}.')
        for basis in (REFERENCE, STAGE2):
            row = nbase.get((identifier, basis), {}); other = gbase[identifier, basis]
            require(row.get('status') == 'passed' and row.get('native_source_sha256') == SCIENCE_HASH
                    and shared.finite(row.get('energy_hartree'))
                    and abs(row['energy_hartree'] - other['energy_hartree']) <= 1e-7,
                    f'Invalid baseline native parity {identifier}/{basis}.')
            require(row['nao'] == other['nao'] == nbase[identifier, REFERENCE]['nao']
                    and row['primitive_nao'] == other['primitive_nao'] == nbase[identifier, REFERENCE]['primitive_nao'],
                    f'Baseline NNAO AO/primitive counts differ for {identifier}.')
    new_native = shared.index_results(native); new_gpu = shared.index_results(gpu4)
    require(set(identifier for identifier, _ in new_native) <= identifiers
            and set(identifier for identifier, _ in new_gpu) <= identifiers
            and all(basis == PILOT for _, basis in (*new_native, *new_gpu)),
            'New results may contain only the selected pilot-best975 cases.')
    profile = gpu4.get('model_profiles', {}).get(PILOT, {})
    if any(row.get('status') == 'passed' for row in new_gpu.values()):
        require(profile.get('checkpoint_profile') == 'pilot-best975'
                and profile.get('checkpoint_epoch') == 975
                and profile.get('model_training_structure_count') == 24
                and profile.get('checkpoint_sha256') == PILOT_CHECKPOINT
                and profile.get('native_source_sha256') == SCIENCE_HASH,
                'Independent GPU4PySCF frozen model profile differs.')
    for records in (new_native, new_gpu):
        for (identifier, _), row in records.items():
            if row.get('status') == 'passed':
                initial = nbase[identifier, REFERENCE]
                require(row.get('nao') == initial['nao'] and row.get('primitive_nao') == initial['primitive_nao'],
                        f'{identifier}: NNAO AO/primitive counts differ from the fixed baseline pool.')
                if records is new_native:
                    require(row.get('native_source_sha256') == SCIENCE_HASH,
                            f'{identifier}: native case source differs.')


def collect_reports(bn, bg, native, gpu4, structures):
    merged_native = dict(native, results=bn['results'] + native.get('results', []))
    merged_gpu = dict(gpu4, results=bg['results'] + gpu4.get('results', []))
    rows = shared.collect(merged_native, merged_gpu, structures)
    nbase = shared.index_results(bn)
    profiles = {REFERENCE: (0, 0), STAGE2: (41, 200), PILOT: (975, 24)}
    by_case = {(row['structure_id'], row['basis']): row for row in rows}
    for row in rows:
        row.pop('reference_to_epoch41_gain_millihartree')
        row.update(checkpoint_epoch=None, model_training_structure_count=None,
                   energy_reduction_vs_nnao_reference_millihartree=None,
                   pilot975_reduction_vs_stage2epoch41_millihartree=None)
        if row['basis'] in profiles:
            row['checkpoint_epoch'], row['model_training_structure_count'] = profiles[row['basis']]
            if row['basis'] == STAGE2:
                original = nbase[row['structure_id'], STAGE2]
                row['native_status'] = 'passed'; row['native_energy_hartree'] = original['energy_hartree']
                row['native_energy_error_hartree'] = abs(row['energy_hartree'] - original['energy_hartree'])
            reference = by_case[row['structure_id'], REFERENCE]['energy_hartree']
            if row['energy_hartree'] is not None and reference is not None:
                row['energy_reduction_vs_nnao_reference_millihartree'] = 1000 * (reference - row['energy_hartree'])
            if row['basis'] == PILOT and row['energy_hartree'] is not None:
                stage2 = by_case[row['structure_id'], STAGE2]['energy_hartree']
                if stage2 is not None:
                    row['pilot975_reduction_vs_stage2epoch41_millihartree'] = 1000 * (stage2 - row['energy_hartree'])
    return rows


def plots(rows, identifiers, output):
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='nnao-pilot-report-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    import numpy as np
    by_case = {(row['structure_id'], row['basis']): row for row in rows}
    values = np.array([[by_case[identifier, basis]['above_cc_pvtz_millihartree']
                       if shared.finite(by_case[identifier, basis]['above_cc_pvtz_millihartree']) else np.nan
                       for identifier in identifiers] for basis in BASES])
    limit = max(10., float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else 10.
    norm = SymLogNorm(linthresh=10., vmin=-limit, vmax=limit)
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('#e6e6e6')
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 11.2), layout='constrained',
                             gridspec_kw={'height_ratios': [3.7, 1.3]})
    heat = axes[0].imshow(np.ma.masked_invalid(values), aspect='auto', norm=norm, cmap=cmap)
    labels = [basis if basis not in (STAGE2, PILOT) else
              ('NNAO: train200 / epoch41' if basis == STAGE2 else 'NNAO: train24 / best975') for basis in BASES]
    axes[0].set_yticks(range(len(BASES)), labels, fontsize=8)
    axes[0].set_xticks(range(len(identifiers)), identifiers, rotation=35, ha='right', fontsize=7)
    axes[0].set_title('Frozen models: same-geometry cc-pVTZ finite-basis DF-RHF reference')
    for i, basis in enumerate(BASES):
        for j, identifier in enumerate(identifiers):
            value = values[i, j]
            text = (f'{value:.1f}' if abs(value) < 1000 else f'{value:.0f}') if np.isfinite(value) else (
                'fail' if by_case[identifier, basis]['comparison_status'] == 'failed' else 'n/a')
            color = 'white' if np.isfinite(value) and (norm(value) < .15 or norm(value) > .85) else 'black'
            axes[0].text(j, i, text, ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(heat, ax=axes[0], label='Energy difference / mHa (signed symmetric-log colors)', shrink=.9)
    x = np.arange(len(identifiers))
    for offset, basis, label, color in ((-.18, STAGE2, 'train200 / epoch41', '#4383aa'),
                                       (.18, PILOT, 'train24 / best975', '#217c74')):
        gains = [by_case[identifier, basis]['energy_reduction_vs_nnao_reference_millihartree']
                 if shared.finite(by_case[identifier, basis]['energy_reduction_vs_nnao_reference_millihartree']) else np.nan
                 for identifier in identifiers]
        axes[1].bar(x + offset, gains, width=.36, label=label, color=color)
    axes[1].set_xticks(x, identifiers, rotation=35, ha='right', fontsize=7)
    axes[1].set_ylabel('Energy reduction / mHa'); axes[1].axhline(0., color='gray', linewidth=.8)
    axes[1].set_title('Energy lowering from the same initial contractions; positive means lower energy')
    axes[1].margins(y=.2); axes[1].legend(frameon=False, fontsize=8)
    for suffix in ('png', 'pdf'): fig.savefig(output / f'energy-gap.{suffix}', dpi=180)
    plt.close(fig)

    trained = [by_case[identifier, PILOT] for identifier in identifiers]
    fig, axes = plt.subplots(2, 2, figsize=(13., 8.3), layout='constrained')
    def panel(axis, items, title, status):
        width = .22 if len(items) == 3 else .36
        for i, (field, label, color) in enumerate(items):
            values = [row[field] if row[status] == 'passed' and shared.finite(row[field]) else np.nan
                      for row in trained]
            axis.bar(x + (i - (len(items) - 1) / 2) * width, values, width, label=label, color=color)
        axis.set_title(title, fontsize=10); axis.set_ylabel('Seconds')
        axis.set_xticks(x, identifiers, rotation=40, ha='right', fontsize=7)
        axis.margins(y=.25); axis.legend(frameon=False, fontsize=7.5)
    panel(axes[0, 0], [('native_nn_cold_seconds', 'First call incl. JIT', '#3776a8'),
          ('native_nn_warm_seconds', 'Warm mean', '#70b5b3')], 'MACE prediction: frozen train24 / best975', 'native_status')
    panel(axes[0, 1], [('native_basis_binding_to_cpu_seconds', 'Bind + send to CPU', '#80759b'),
          ('native_integral_seconds', 'CPU S/H/DF + whiten/unpack', '#b08451'),
          ('native_host_device_transfer_seconds', 'CPU-to-SCF transfer', '#70b5b3')],
          'Integral preparation and explicit transfers', 'native_status')
    panel(axes[1, 0], [('native_scf_cold_seconds', 'First call incl. JIT', '#3776a8'),
          ('native_scf_warm_seconds', 'Warm mean (2 calls)', '#70b5b3')],
          'JAX SCF + validation diagnostics', 'native_status')
    panel(axes[1, 1], [('gpu4_scf_cold_seconds', 'First kernel incl. DF setup', '#3776a8'),
          ('gpu4_scf_warm_seconds', 'Warm kernel (1 call)', '#70b5b3')], 'GPU4PySCF SCF kernel', 'gpu4_status')
    fig.suptitle('New best975 inference only: distinct timing scopes, not a speedup comparison', fontsize=12)
    for suffix in ('png', 'pdf'): fig.savefig(output / f'timing.{suffix}', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--gpu4', type=Path, required=True)
    parser.add_argument('--baseline-native', type=Path, default=DIRECTORY / 'job-95106/native/forward.json')
    parser.add_argument('--baseline-gpu4', type=Path, default=DIRECTORY / 'job-95106/gpu4/comparison.json')
    parser.add_argument('--structures', type=Path, default=DIRECTORY / 'test-selection.jsonl')
    parser.add_argument('--output-dir', type=Path, default=DIRECTORY / 'report-pilot-best975')
    args = parser.parse_args(argv)
    require(args.output_dir.resolve() != (DIRECTORY / 'report').resolve(), 'The prior report may not be overwritten.')
    require(sha256(args.structures) == STRUCTURES_HASH, 'The selected structure bytes differ from the verified ten cases.')
    for name, path in (('native', args.baseline_native), ('gpu4', args.baseline_gpu4)):
        require(sha256(path) == BASELINE_HASHES[name], f'{name} baseline artifact bytes changed.')
    bn = json.loads(args.baseline_native.read_text()); bg = json.loads(args.baseline_gpu4.read_text())
    native = json.loads(args.native.read_text()); gpu4 = json.loads(args.gpu4.read_text())
    validate_reports(bn, bg, native, gpu4)
    structures = [json.loads(line) for line in args.structures.read_text().splitlines() if line.strip()]
    identifiers = [row['structure_id'] for row in structures]
    rows = collect_reports(bn, bg, native, gpu4, structures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_case = {(row['structure_id'], row['basis']): row for row in rows}
    pilot = [by_case[identifier, PILOT] for identifier in identifiers]
    good = [row for row in pilot if row['comparison_status'] == 'passed']
    parity = [row['native_energy_error_hartree'] for row in good]
    levels = {}
    for basis in BASES[:12]:
        differences = [row['energy_hartree'] - by_case[row['structure_id'], basis]['energy_hartree'] for row in good]
        levels[basis] = dict(compared_structures=len(differences),
            lower_energy=sum(value < -1e-9 for value in differences),
            higher_energy=sum(value > 1e-9 for value in differences),
            equal_within_1e_minus9_hartree=sum(abs(value) <= 1e-9 for value in differences))
    summary = dict(status='complete' if len(good) == 10 else 'partial',
        structure_count=10, combined_basis_count=15, combined_passed_energy_cases=sum(
            row['comparison_status'] == 'passed' for row in rows),
        pilot_passed=len(good), pilot_failed=sum(row['comparison_status'] == 'failed' for row in pilot),
        pilot_pending=sum(row['comparison_status'] == 'pending' for row in pilot),
        models={REFERENCE: dict(training_structures=0, epoch=0, note='Initial contractions'),
                STAGE2: dict(training_structures=200, epoch=41, checkpoint_sha256=STAGE2_CHECKPOINT),
                PILOT: dict(training_structures=24, epoch=975, checkpoint_sha256=PILOT_CHECKPOINT)},
        mean_pilot_gain_vs_reference_millihartree=shared.mean(
            row['energy_reduction_vs_nnao_reference_millihartree'] for row in good),
        mean_pilot_gain_vs_stage2epoch41_millihartree=shared.mean(
            row['pilot975_reduction_vs_stage2epoch41_millihartree'] for row in good),
        mean_pilot_above_ccpvtz_millihartree=shared.mean(row['above_cc_pvtz_millihartree'] for row in good),
        pilot_lower_than_reference=sum(row['energy_reduction_vs_nnao_reference_millihartree'] > 1e-6 for row in good),
        pilot_lower_than_stage2epoch41=sum(row['pilot975_reduction_vs_stage2epoch41_millihartree'] > 1e-6 for row in good),
        max_native_gpu4_energy_error_hartree=max(parity) if parity else None,
        traditional_comparisons=levels, scientific_source_sha256=SCIENCE_HASH,
        structures_sha256=STRUCTURES_HASH,
        input_hashes={name: sha256(path) for name, path in (('native', args.native), ('gpu4', args.gpu4),
            ('baseline_native', args.baseline_native), ('baseline_gpu4', args.baseline_gpu4))})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    def display(value): return f'{value:.2f}' if shared.finite(value) else 'n/a'
    error = summary['max_native_gpu4_energy_error_hartree']
    parity_text = f'{error:.3e} Ha' if shared.finite(error) else 'n/a'
    lines = ['# 旧24体系最佳快照的独立 HF 测试', '',
        '**本次只使用旧24体系共享 MACE 的 best.npz（epoch975），冻结推理，没有继续训练或更新参数。** '
        '它与从头训练的200体系模型（epoch41）是两套不同权重，不能称为200体系模型的1000轮结果。', '',
        '复用已完成的作业95106中12个传统基组、初始NNAO和200体系epoch41结果；原始JSON字节哈希均核验。'
        '新测量仅包含10个pilot-best975 native／GPU4PySCF同基组验证，共合并为15基组×10结构。', '',
        f'旧best975有效能量 **{len(good)}/10**，{summary["pilot_failed"]}项失败、{summary["pilot_pending"]}项尚无结果；'
        '失败或后端不一致的计算不计分。'
        f'Native／GPU4PySCF同基组最大误差为 {parity_text}。', '',
        '结构仍为先前10个选定测试体系，结构ID、坐标及整体分子式排除了200训练／50验证／24pilot及旧苯胺。'
        '不保证片段、化学图或反应关联隔离。统一球谐、全电子、float64、DF-RHF、def2-universal-jkfit；'
        '几何、SCF阈值和hcore初始化不变。NNAO AO与primitive池规模逐体系核对一致。', '',
        '| 结构 | NNAO AO / primitive AO | best975 − cc-pVTZ / mHa | 相对初始降低 / mHa | 相对200体系epoch41降低 / mHa | 状态 |',
        '|---|---:|---:|---:|---:|---|']
    for row in pilot:
        lines.append(f'| {row["structure_id"]} | {row["nao"]} / {row["primitive_nao"]} | '
            f'{display(row["above_cc_pvtz_millihartree"])} | '
            f'{display(row["energy_reduction_vs_nnao_reference_millihartree"])} | '
            f'{display(row["pilot975_reduction_vs_stage2epoch41_millihartree"])} | {row["comparison_status"]} |')
    lines += ['', f'仅按有效独立测试结果，best975相对初始平均降低 **{display(summary["mean_pilot_gain_vs_reference_millihartree"])} mHa**；'
        f'相对200体系epoch41平均降低 **{display(summary["mean_pilot_gain_vs_stage2epoch41_millihartree"])} mHa**。'
        '正值表示旧best975能量更低，负值表示更高。这些数值来自本次测试，未使用旧24训练均值作为得分。', '',
        '按逐体系实际能量，best975低于传统基组的结构数为：' + '；'.join(
            f'{basis} {levels[basis]["lower_energy"]}/{len(good)}' for basis in ('minao', '3-21g', '6-31g', 'cc-pvdz')) + '。', '',
        '参照为同几何cc-pVTZ有限基组DF-HF能量，非CBS、非GMTKN55高精度标签；不是完整GMTKN55或WTMAD2评估。'
        '不能由能量降低推断力、BSSE或能力上限。此次没有测试旧last1000，也没有解释旧训练epoch976的N2跳变。', '',
        '计时图只报告新best975前向：基组绑定／到CPU、CPU积分及DF白化解包、到SCF设备传输分开记录；'
        'MACE和JAX首调用包含编译，native solve还含能量重构、S最小特征值、commutator及raw-Phi验收。'
        'GPU4只对mf.kernel计时，其诊断在计时外；冷热SCF都独立从hcore开始，不计算后端speedup。'
        '新旧两次硬件均核对为Tesla V100-SXM2-32GB。', '',
        '![15基组有限参照能量差](energy-gap.png)', '', '![仅新best975的独立计时](timing.png)', '',
        f'新测量：`{args.native.resolve()}`、`{args.gpu4.resolve()}`；基线：'
        f'`{args.baseline_native.resolve()}`、`{args.baseline_gpu4.resolve()}`。']
    (args.output_dir / 'README.md').write_text('\n'.join(lines) + '\n')
    plots(rows, identifiers, args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('traditional_comparisons', 'models')}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
