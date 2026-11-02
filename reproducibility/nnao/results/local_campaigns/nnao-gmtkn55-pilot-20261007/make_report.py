"""Summarize recorded NNAO pilot outcomes without running scientific code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SYSTEMS = Path('/Volumes/TF/GradSCF_GMTKN55/selections/hf_pilot_20261007/systems.csv')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def fmt(value, spec='.3e'):
    value = number(value)
    return '—' if value is None else format(value, spec)


def read_json(path, inputs):
    path = Path(path).resolve()
    content = json.loads(path.read_text())
    inputs[str(path)] = dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
    return content


def records_from(content):
    """Accept a suite summary, one result, a result list, or an ID mapping."""
    if isinstance(content, list):
        rows = content
    elif isinstance(content, dict) and 'results' in content:
        return records_from(content['results'])
    elif isinstance(content, dict) and 'structure_id' in content:
        rows = [content]
    elif isinstance(content, dict):
        rows = [dict(value, structure_id=key) for key, value in content.items()
                if isinstance(value, dict)]
    else:
        raise ValueError('JSON must contain result records.')
    found = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get('structure_id'):
            raise ValueError('A result has no structure_id.')
        identifier = row['structure_id']
        if identifier in found:
            raise ValueError(f'Duplicate result ID: {identifier}')
        found[identifier] = row
    return found


def load_validation(directory, inputs):
    directory = Path(directory).resolve()
    path = directory / 'summary.json'
    summary = read_json(path, inputs) if path.is_file() else {}
    records = records_from(summary) if summary else {}
    origins = {key: str(path) for key in records}
    # Per-structure files are the latest persisted outcome; retain input hashes.
    seen = set()
    for child in sorted(directory.glob('*/result.json')):
        result = read_json(child, inputs)
        identifier = result.get('structure_id')
        if not identifier or identifier in seen:
            raise ValueError(f'Missing/duplicate structure ID in {child}')
        seen.add(identifier)
        records[identifier] = result
        origins[identifier] = str(child.resolve())
    return summary, records, origins


def checked_outcome(record, expected_steps):
    errors = []
    if not record:
        return 'missing', ['未找到结果记录']
    if record.get('status') != 'passed':
        errors.append(record.get('error') or f"原始状态={record.get('status', 'missing')}")
    if record.get('implicit_gradient_finite') is not True:
        errors.append('初始 implicit 梯度未确认有限')
    for key, label in [('coefficient_directional_check', '收缩输出 FD'),
                       ('parameter_directional_check', 'MACE 参数 FD')]:
        check = record.get(key, {})
        if check.get('passed') is not True or number(check.get('absolute_error')) is None:
            errors.append(f'{label} 缺失或未通过')
    steps = number(record.get('accepted_adam_steps'))
    if steps is None or steps < expected_steps or record.get('adam_ran') is not True:
        errors.append(f'未完成 {expected_steps} 步 Adam')
    decrease = number(record.get('energy_decrease_hartree'))
    if decrease is None or decrease < -1e-10:
        errors.append('降能缺失或为负')
    initial = number(record.get('initial_energy_hartree'))
    final = number(record.get('final_energy_hartree'))
    if initial is None or final is None:
        errors.append('初始/最终能量缺失')
    elif decrease is not None and abs((initial - final) - decrease) > 1e-9:
        errors.append('记录的降能与端点能量不一致')
    history = record.get('adam_history', [])
    if len(history) < expected_steps + 1:
        errors.append('Adam 轨迹不完整')
    for point in [record.get('initial_scf', {}), *history]:
        if point.get('converged') is not True:
            errors.append('SCF 收敛记录缺失或未通过')
            break
        if (number(point.get('orbital_residual')) is None
                or point['orbital_residual'] > 1e-7
                or number(point.get('min_overlap_eigenvalue')) is None
                or point['min_overlap_eigenvalue'] < 1e-8
                or number(point.get('reconstruction_error')) is None
                or point['reconstruction_error'] > 1e-9):
            errors.append('SCF 数值质量未通过原脚本准入阈值')
            break
    return ('passed' if not errors else 'failed'), errors


def summarize(values, operation=max):
    finite = [value for item in values if (value := number(item)) is not None]
    return operation(finite) if finite else None


def duration(records):
    values = [number(record.get('elapsed_seconds')) for record in records.values()]
    return dict(seconds=sum(value for value in values if value is not None),
                records_with_time=sum(value is not None for value in values),
                records=len(values))


def source_snapshots(records, config):
    snapshots = {}
    for record in records.values():
        for name, evidence in record.get('imported_sources', {}).items():
            key = (name, evidence.get('path'), evidence.get('sha256'))
            snapshots[key] = dict(module=name, **evidence)
    return dict(config=config, imported_sources=list(snapshots.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-dir', type=Path, default=HERE / 'job-93084/validation')
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--systems-csv', type=Path, default=DEFAULT_SYSTEMS)
    parser.add_argument('--supplement-json', type=Path)
    parser.add_argument('--override-summary', type=Path)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args(argv)
    inputs = {}
    with args.systems_csv.open(newline='') as stream:
        systems = list(csv.DictReader(stream))
    identifiers = [row['structure_id'] for row in systems]
    if len(systems) != 24 or len(set(identifiers)) != 24:
        raise ValueError('The selected pilot must contain exactly 24 unique records.')
    inputs[str(args.systems_csv.resolve())] = dict(path=str(args.systems_csv.resolve()),
        sha256=sha256(args.systems_csv), bytes=args.systems_csv.stat().st_size)
    base_summary, original, origins = load_validation(args.validation_dir, inputs)
    selected = dict(original)
    overrides = {}
    override_summary = {}
    if args.override_summary:
        override_summary = read_json(args.override_summary, inputs)
        overrides = records_from(override_summary)
        unknown = set(overrides) - set(identifiers)
        if unknown:
            raise ValueError(f'Override contains unselected IDs: {sorted(unknown)}')
        for identifier, record in overrides.items():
            selected[identifier] = record
            origins[identifier] = str(args.override_summary.resolve())
    supplement_doc = read_json(args.supplement_json, inputs) if args.supplement_json else {}
    supplements = records_from(supplement_doc) if supplement_doc else {}
    unknown = (set(selected) | set(supplements)) - set(identifiers)
    if unknown:
        raise ValueError(f'Results contain unselected IDs: {sorted(unknown)}')
    preparation_dir = args.preparation_dir or args.validation_dir.parent / 'preparation'
    if preparation_dir.is_dir():
        preparation_summary, prepared, _ = load_validation(preparation_dir, inputs)
    else:
        preparation_summary, prepared = {}, {}
    expected_steps = int(base_summary.get('config', {}).get('adam_steps', 5))
    if expected_steps != 5:
        raise ValueError('This report is scoped to the agreed five-step pilot.')

    metrics, direct_points, supplement_failures = [], [], []
    for system in systems:
        identifier = system['structure_id']
        record, extra = selected.get(identifier, {}), supplements.get(identifier, {})
        outcome, errors = checked_outcome(record, expected_steps)
        final_check = extra.get('final_parameter_directional_check', {})
        if extra and (extra.get('status') == 'failed' or final_check.get('passed') is False):
            errors.append('补充最终参数 FD/精确 direct 对照未通过')
            supplement_failures.append(identifier)
            outcome = 'failed'
        points = extra.get('exact_direct_comparisons', [])
        for point in points:
            if number(point.get('df_minus_direct_hartree')) is not None:
                direct_points.append(dict(structure_id=identifier, **point))
        history = record.get('adam_history', [])
        quality = [record.get('initial_scf', {}), *history, extra.get('final_scf', {})]
        metric = dict(structure_id=identifier, name=system['name'], group=system['group'],
            status=outcome, recorded_status=record.get('status', 'missing'),
            original_status=original.get(identifier, {}).get('status', 'missing'),
            overridden=identifier in overrides, nao=record.get('nao'),
            primitive_nao=record.get('primitive_nao'), natoms=int(system['natoms']),
            initial_energy_hartree=number(record.get('initial_energy_hartree')),
            final_energy_hartree=number(record.get('final_energy_hartree')),
            energy_decrease_millihartree=(1000 * record['energy_decrease_hartree']
                if number(record.get('energy_decrease_hartree')) is not None else None),
            coefficient_fd_absolute_error=number(record.get('coefficient_directional_check', {}).get('absolute_error')),
            parameter_fd_absolute_error=number(record.get('parameter_directional_check', {}).get('absolute_error')),
            final_parameter_fd_absolute_error=number(final_check.get('absolute_error')),
            final_parameter_fd_passed=final_check.get('passed'),
            accepted_adam_steps=record.get('accepted_adam_steps'),
            maximum_orbital_residual=summarize(point.get('orbital_residual') for point in quality),
            minimum_overlap_eigenvalue=summarize((point.get('min_overlap_eigenvalue') for point in quality), min),
            maximum_reconstruction_error_hartree=summarize(point.get('reconstruction_error') for point in quality),
            maximum_scf_cycles=summarize(point.get('scf_cycles') for point in quality),
            validation_elapsed_seconds=number(record.get('elapsed_seconds')),
            supplement_elapsed_seconds=number(extra.get('elapsed_seconds')),
            direct_point_count=len(points),
            maximum_absolute_df_minus_direct_hartree=summarize(abs(point['df_minus_direct_hartree'])
                for point in points if number(point.get('df_minus_direct_hartree')) is not None),
            source_result=origins.get(identifier), source_signature=record.get('signature'),
            cache_path=record.get('cache_path'), cache_sha256=record.get('cache_sha256'),
            integral_signature=record.get('integral_signature'),
            error='; '.join(dict.fromkeys(errors)), orbital_stability='not_assessed')
        metrics.append(metric)

    successful = [row for row in metrics if row['status'] == 'passed']
    original_failures = [dict(structure_id=key, result=record)
                         for key, record in original.items() if record.get('status') != 'passed']
    aggregates = dict(
        maximum_coefficient_fd_absolute_error=summarize(row['coefficient_fd_absolute_error'] for row in successful),
        maximum_parameter_fd_absolute_error=summarize(row['parameter_fd_absolute_error'] for row in successful),
        maximum_final_parameter_fd_absolute_error=summarize(row['final_parameter_fd_absolute_error'] for row in metrics),
        minimum_energy_decrease_millihartree=summarize((row['energy_decrease_millihartree'] for row in successful), min),
        maximum_energy_decrease_millihartree=summarize(row['energy_decrease_millihartree'] for row in successful),
        maximum_orbital_residual=summarize(row['maximum_orbital_residual'] for row in successful),
        minimum_overlap_eigenvalue=summarize((row['minimum_overlap_eigenvalue'] for row in successful), min),
        maximum_reconstruction_error_hartree=summarize(row['maximum_reconstruction_error_hartree'] for row in successful),
        direct_comparison_points=len(direct_points),
        maximum_absolute_df_minus_direct_hartree=summarize(abs(point['df_minus_direct_hartree']) for point in direct_points))
    def execution_key(identifier, record):
        return (identifier, record.get('slurm_job_id'), record.get('signature'),
                record.get('output_dir'), record.get('elapsed_seconds'))
    original_execution_keys = {execution_key(key, record) for key, record in original.items()}
    new_retries = {key: record for key, record in overrides.items()
                   if execution_key(key, record) not in original_execution_keys}
    times = dict(preparation=duration(prepared), original_validation=duration(original),
                 retry_validation=duration(new_retries), supplement=duration(supplements))
    times['total_recorded_work_seconds'] = sum(value['seconds'] for value in times.values())
    times['is_scheduler_walltime'] = False
    verification = dict(expected_records=24, passed=len(successful),
        failed=sum(row['status'] == 'failed' for row in metrics),
        missing=sum(row['status'] == 'missing' for row in metrics),
        overall_passed=len(successful) == 24,
        original_summary_counts={key: base_summary.get(key) for key in ('passed', 'failed', 'exit_code')},
        methods_scope='Per-structure independent MACE model; fixed-exponent all-electron spherical SZP; RHF/DF; implicit backward; five Adam steps.',
        shared_model_training=False, orbital_stability_assessed=False,
        numeric_adjoint_residual_exposed=False, aggregates=aggregates,
        native_library_sha256=sorted({record['native_library_sha256'] for record in prepared.values()
                                      if record.get('native_library_sha256')}),
        recorded_work_times=times, inputs=list(inputs.values()),
        recorded_source_snapshots=dict(original=source_snapshots(original, base_summary.get('config', {})),
            override=source_snapshots(overrides, override_summary.get('config', {}) if isinstance(override_summary, dict) else {}),
            preparation=source_snapshots(prepared, preparation_summary.get('config', {})),
            supplement=source_snapshots(supplements, {key: value for key, value in supplement_doc.items()
                if key != 'results'} if isinstance(supplement_doc, dict) else {})),
        original_failed_history=original_failures,
        replaced_records=[dict(structure_id=key, original_result=original.get(key), replacement_result=record)
                          for key, record in overrides.items()],
        supplement_failed_ids=supplement_failures, direct_comparisons=direct_points,
        metrics=metrics, report_script_sha256=sha256(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (args.output_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    config = base_summary.get('config', {})
    example = next((selected[row['structure_id']] for row in successful), {})
    scf_config = example.get('scf_config', {})
    implicit_config = example.get('implicit_config', {})
    hardware = sorted({(record.get('hostname', '未记录'), ', '.join(record.get('device_kinds', [])),
                       record.get('jax_version', '未记录'), record.get('dtype', '未记录'))
                      for record in selected.values()})
    lines = ['# GMTKN55 NNAO 小样本 RHF 验证', '',
        f"**验证链路通过 {len(successful)}/24；总体{'通过' if len(successful) == 24 else '未通过'}。**",
        f"原始运行记录为 {base_summary.get('passed', '未记录')} 通过、{base_summary.get('failed', '未记录')} 失败；"
        f"使用 {len(overrides)} 条显式重跑替换，原失败记录保留在 `verification.json`。", '',
        '## 实际方法与范围', '',
        '- GMTKN55 v2 的 24 条 CHNOF、中性、spin=0 记录；名称与顺序来自保存的 `systems.csv`。',
        '- RHF，`szp663_direct/core6`，球谐 AO；几何、primitive 指数和内层固定，MACE 预测可学习壳层的收缩系数。',
        '- CPU native 生成固定 primitive overlap、hcore 和 packed DF factors；GPU/JAX 执行收缩、JK、SCF 与 implicit backward；未生成完整四阶 ERI。',
        f"- DF 辅助基组 `{config.get('auxbasis', '未记录')}`，Adam 学习率 `{config.get('learning_rate', '未记录')}`，seed `{config.get('seed', '未记录')}`，每体系 5 步。",
        f"- 主运行 SCF 配置：`{json.dumps(scf_config, sort_keys=True)}`；implicit 配置：`{json.dumps(implicit_config, sort_keys=True)}`。替换记录的各自配置保存在来源快照中。",
        '- 每个结构独立初始化并优化一套 MACE 参数，统一 CHNOF 词表；这次不是共享模型训练，也不证明跨体系泛化。',
        '- 初始收缩输出及初始 MACE 参数使用四阶中心有限差分，每个扰动重新收敛 SCF。',
        f"- FD 步长 `{config.get('fd_step', '未记录')}`；验收为 `atol={config.get('fd_atol', '未记录')}` 加 `rtol={config.get('fd_rtol', '未记录')}` 乘参考导数绝对值。", '',
        '硬件/软件记录：' + '；'.join(f'{node} / {gpu} / JAX {version} / {dtype}' for node, gpu, version, dtype in hardware) + '。', '',
        '## 结果', '',
        f"- 初始输出方向 AD–FD 最大绝对误差：**{fmt(aggregates['maximum_coefficient_fd_absolute_error'])} Ha/方向单位**。",
        f"- 初始 MACE 参数方向 AD–FD 最大绝对误差：**{fmt(aggregates['maximum_parameter_fd_absolute_error'])} Ha/方向单位**。",
        f"- 通过记录的 5 步降能范围：**{fmt(aggregates['minimum_energy_decrease_millihartree'], '.6f')}–{fmt(aggregates['maximum_energy_decrease_millihartree'], '.6f')} mHa**。",
        f"- 通过记录的最大轨道残差：`{fmt(aggregates['maximum_orbital_residual'])}`；最小 overlap 特征值：`{fmt(aggregates['minimum_overlap_eigenvalue'])}`；最大能量重构差：`{fmt(aggregates['maximum_reconstruction_error_hartree'])}` Ha。", '',
        '| ID | 体系 | AO / primitive AO | 初始 E / Ha | 最终 E / Ha | 降能 / mHa | 输出 FD 误差 | 参数 FD 误差 | Adam 步 | 验证秒 | 结果 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in metrics:
        state = {'passed': '通过', 'failed': '未通过', 'missing': '缺失'}[row['status']]
        suffix = '（重跑）' if row['overridden'] else ''
        lines.append(f"| {row['structure_id']} | {row['name']} | {row['nao'] or '—'} / {row['primitive_nao'] or '—'} | "
            f"{fmt(row['initial_energy_hartree'], '.12f')} | {fmt(row['final_energy_hartree'], '.12f')} | "
            f"{fmt(row['energy_decrease_millihartree'], '.6f')} | {fmt(row['coefficient_fd_absolute_error'])} | "
            f"{fmt(row['parameter_fd_absolute_error'])} | {row['accepted_adam_steps'] if row['accepted_adam_steps'] is not None else '—'} | "
            f"{fmt(row['validation_elapsed_seconds'], '.2f')} | {state}{suffix} |")
    failures = [row for row in metrics if row['status'] != 'passed']
    if failures:
        lines.extend(['', '未通过记录：', ''])
        lines.extend(f"- `{row['structure_id']}`：{row['error']}。" for row in failures)
    lines.extend(['', '## 补充最终梯度与 direct 对照', ''])
    if supplements:
        final_checks = [(key, row.get('final_parameter_directional_check', {})) for key, row in supplements.items()]
        checked = [(key, check) for key, check in final_checks if number(check.get('absolute_error')) is not None]
        lines.extend([f"补充最终参数 FD 记录 {len(checked)} 条，其中明确通过 {sum(check.get('passed') is True for _, check in checked)} 条；"
            f"最大绝对误差 `{fmt(aggregates['maximum_final_parameter_fd_absolute_error'])}` Ha/方向单位。",
            f"精确 shell-direct 与 DF 的能量对照共 {len(direct_points)} 点，最大 `|E_DF-E_direct|` 为 "
            f"**{fmt(aggregates['maximum_absolute_df_minus_direct_hartree'])} Ha**。该差值是 Hamiltonian 近似差异，不等于 AD 误差。", '',
            '| ID | 最终参数 FD 误差 | FD 状态 | direct 点数 | 最大 DF/direct 能量差 / Ha |', '|---|---:|---|---:|---:|'])
        for row in metrics:
            if row['structure_id'] in supplements:
                state = '通过' if row['final_parameter_fd_passed'] is True else '未通过/未记录'
                lines.append(f"| {row['structure_id']} | {fmt(row['final_parameter_fd_absolute_error'])} | {state} | "
                    f"{row['direct_point_count']} | {fmt(row['maximum_absolute_df_minus_direct_hartree'])} |")
    else:
        lines.append('未提供补充结果；最终参数 FD、精确 direct 对照均未评估。')
    lines.extend(['', '## 时间与未验证边界', '',
        f"已记录 CPU 准备耗时加总 `{fmt(times['preparation']['seconds'], '.2f')}` s "
        f"（{times['preparation']['records_with_time']}/{times['preparation']['records']} 条有时间字段）；"
        f"原始科学验证 `{fmt(times['original_validation']['seconds'], '.2f')}` s；"
        f"重跑 `{fmt(times['retry_validation']['seconds'], '.2f')}` s；"
        f"补充检查 `{fmt(times['supplement']['seconds'], '.2f')}` s。",
        f"以上已记录工作合计 `{fmt(times['total_recorded_work_seconds'], '.2f')}` s，包含失败尝试；"
        '这是逐记录计算时间之和，包含编译，**不是 Slurm 作业 walltime**，未记录的加载、排队、构建或诊断时间不在内。', '',
        '- SCF 收敛与轨道残差只验证驻点；RHF internal、spin-breaking Hessian 稳定性本轮未评估，F₂/N₂也不能称为已验证稳定基态。',
        '- 现有 pilot 结果未暴露数值 adjoint/GMRES 残差；有限梯度和 FD 一致性不能替代该残差的直接记录。',
        '- 没有训练/验证/测试家族划分、共享参数训练、离子/UHF、DFT、力或 TDDFT 验证。',
        '- 不将降能解释为达到 DZ/TZ 精度，也不将初始少数 FD 误差推广为所有最终参数的精度。', '',
        '## 来源与可重建性', '',
        f"- 结构名称清单：`{args.systems_csv.resolve()}`。",
        f"- 主结果：`{args.validation_dir.resolve()}`。",
        '- `metrics.csv` 保存 24 条结果与单位化指标；`verification.json` 保存每个读取文件的 SHA-256、记录的远程导入路径/源码哈希、原失败历史和替换记录。',
        '- 输入文件哈希由本报告生成时计算；远程源码/native 库哈希来自运行记录，此脚本没有重新登录或复核远程文件。',
        '- 不将早先从其他安装路径导入代码的运行自动并入主结果。', '', '报告重建命令：', '', '```sh',
        f"python artifacts/nnao-gmtkn55-pilot-20261007/make_report.py --validation-dir '{args.validation_dir.resolve()}'"
        + (f" --supplement-json '{args.supplement_json.resolve()}'" if args.supplement_json else '')
        + (f" --override-summary '{args.override_summary.resolve()}'" if args.override_summary else ''), '```', ''])
    (args.output_dir / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), passed=len(successful),
        expected=24, overall_passed=len(successful) == 24, aggregates=aggregates,
        recorded_work_times=times), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
