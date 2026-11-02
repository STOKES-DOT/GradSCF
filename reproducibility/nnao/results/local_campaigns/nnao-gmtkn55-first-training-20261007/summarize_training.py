"""Summarize a consistent accepted-training checkpoint without importing JAX."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        result = json.loads(str(saved['metadata'].item()))
        parameters = np.asarray(saved['parameters'])
        gradient = np.asarray(saved['gradient'])
        if not np.isfinite(parameters).all() or not np.isfinite(gradient).all():
            raise ValueError(f'{path}: nonfinite checkpoint parameters or gradient.')
        result['parameter_sha256'] = hashlib.sha256(parameters.tobytes()).hexdigest()
        result['parameter_count'] = int(parameters.size)
        result['parameter_dtype'] = str(parameters.dtype)
    history = result['history']
    if [row['epoch'] for row in history] != list(range(result['epoch'] + 1)):
        raise ValueError(f'{path}: accepted epoch history is incomplete.')
    if (history[-1]['loss_hartree'] != result['loss_hartree']
            or history[-1]['molecules'] != result['molecules']):
        raise ValueError(f'{path}: loss/diagnostics do not describe the saved parameters.')
    expected = result['identity']['structure_ids']
    for row in history:
        if [item['structure_id'] for item in row['molecules']] != expected:
            raise ValueError(f'{path}: molecules were omitted or reordered.')
        energy = [item['energy_hartree'] for item in row['molecules']]
        if not np.isfinite(energy).all() or not np.isclose(np.mean(energy), row['loss_hartree'], atol=1e-12, rtol=0):
            raise ValueError(f'{path}: loss differs from the equal-weight molecular mean.')
        for item in row['molecules']:
            diagnostics = [item[key] for key in ('orbital_residual', 'min_overlap_eigenvalue',
                                                'reconstruction_error', 'parameter_gradient_norm')]
            if (item.get('converged') is not True or not np.isfinite(diagnostics).all()
                    or item['orbital_residual'] > 1e-7 or item['min_overlap_eigenvalue'] < 1e-8
                    or item['reconstruction_error'] > 1e-9):
                raise ValueError(f'{path}: invalid accepted SCF diagnostics at epoch {row["epoch"]}.')
    return result


def snapshot(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    last = checkpoint(directory / 'last.npz'); best = checkpoint(directory / 'best.npz')
    if metadata['identity'] != last['identity'] or best['identity'] != last['identity']:
        raise ValueError('Training metadata and checkpoints have different identities.')
    if best['epoch'] > last['epoch'] or best['loss_hartree'] != min(row['loss_hartree'] for row in last['history']):
        raise ValueError('Last/best snapshots are inconsistent; fetch them again after an accepted epoch.')
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary is not None:
        if summary.get('identity') != last['identity'] or summary.get('accepted_epoch') != last['epoch']:
            raise ValueError('Training summary and checkpoint snapshots differ; fetch them again.')
        if summary.get('status') == 'complete' and last['epoch'] != metadata['target_epochs']:
            raise ValueError('Complete summary did not reach the requested epoch target.')
    return metadata, last, best, summary


def make_plot(path, last, best, metrics):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gradscf-training-report-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for family in ('PingFang SC', 'Arial Unicode MS', 'Noto Sans CJK SC'):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams['font.family'] = family
            break
        except ValueError:
            continue
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 9), gridspec_kw={'width_ratios': [1.05, 1]})
    initial = last['history'][0]['loss_hartree']
    axes[0].plot([row['epoch'] for row in last['history']],
                 [1000 * (initial - row['loss_hartree']) for row in last['history']],
                 color='#245c9b', linewidth=1.7)
    axes[0].scatter([best['epoch']], [1000 * (initial - best['loss_hartree'])],
                    color='#b34827', zorder=4, label=f"Best: epoch {best['epoch']}")
    axes[0].set(xlabel='Epoch', ylabel='Mean energy decrease (mHa)', title='Shared MACE: equal-weight mean of 24 energies')
    axes[0].grid(alpha=.2); axes[0].legend(loc='lower right')
    decrease = [row['last_decrease_millihartree'] for row in metrics]
    axes[1].barh(np.arange(len(metrics)), decrease,
                 color=['#287c81' if value >= 0 else '#b34827' for value in decrease])
    axes[1].set_yticks(np.arange(len(metrics)), [row['name'] for row in metrics])
    axes[1].invert_yaxis(); axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set(xlabel='Initial minus accepted-checkpoint energy (mHa)',
                title=f"Per-molecule energy decrease: epoch {last['epoch']}")
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle('GMTKN55 pilot / RHF-DF / fixed szp663_direct(core6)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(path, dpi=190); plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, default=root / 'training-93108')
    parser.add_argument('--systems', type=Path, default=Path('/Volumes/TF/GradSCF_GMTKN55/selections/hf_pilot_20261007/systems.csv'))
    parser.add_argument('--output-dir', type=Path, default=root / 'training_report')
    parser.add_argument('--log', type=Path, default=root / 'training-93108.out')
    parser.add_argument('--validation-summary', type=Path, default=root / 'validation-93107/results/summary.json')
    args = parser.parse_args(argv)
    metadata, last, best, summary = snapshot(args.training_dir)
    identity = metadata['identity']; initial = last['history'][0]
    if args.validation_summary.exists() and sha256(args.validation_summary) != identity['validation_summary_sha256']:
        raise ValueError('Training used a different validation summary.')
    validation_text = '训练前验证的详细证据另见 validation_report。'
    if args.validation_summary.exists():
        validation = json.loads(args.validation_summary.read_text())
        validation_text = (f"训练前验证 {validation['passed']}/{len(identity['structure_ids'])} 通过，"
            f"失败 {validation['failed']}；收缩系数/网络参数方向 FD 的最大绝对误差分别为 "
            f"{max(row['coefficient_directional_check']['absolute_error'] for row in validation['results']):.3e} / "
            f"{max(row['parameter_directional_check']['absolute_error'] for row in validation['results']):.3e}。")
    names = {row['structure_id']: row for row in csv.DictReader(args.systems.open())}
    first = {row['structure_id']: row for row in initial['molecules']}
    best_molecules = {row['structure_id']: row for row in best['molecules']}
    metrics = []
    for row in last['molecules']:
        identifier = row['structure_id']; start = first[identifier]; optimal = best_molecules[identifier]
        cache = identity['caches'][identifier]; label = names[identifier]
        metrics.append(dict(structure_id=identifier, name=label['name'], natoms=int(label['natoms']),
            nao=cache['nao'], primitive_nao=cache['primitive_nao'], initial_epoch=0,
            last_epoch=last['epoch'], best_epoch=best['epoch'],
            initial_energy_hartree=start['energy_hartree'], last_energy_hartree=row['energy_hartree'],
            best_checkpoint_energy_hartree=optimal['energy_hartree'],
            last_decrease_millihartree=1000 * (start['energy_hartree'] - row['energy_hartree']),
            best_decrease_millihartree=1000 * (start['energy_hartree'] - optimal['energy_hartree']),
            final_orbital_residual=row['orbital_residual'], final_scf_cycles=row['scf_cycles'],
            final_min_overlap_eigenvalue=row['min_overlap_eigenvalue'],
            final_reconstruction_error=row['reconstruction_error'], cache_sha256=cache['cache_sha256']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    make_plot(args.output_dir / 'training_curve.png', last, best, metrics)
    status = summary.get('status') if summary else 'running'
    status_text = {'complete': '已完成', 'failed': '已停止：训练失败', 'running': '进行中（仅报告已接受的检查点）'}.get(status, str(status))
    decrease = 1000 * (initial['loss_hartree'] - last['loss_hartree'])
    deltas = [row['last_decrease_millihartree'] for row in metrics]
    times = [float(value) for value in re.findall(r';\s*([0-9.]+) s', args.log.read_text())] if args.log.exists() else []
    table = '\n'.join(f"| {row['name']} | {row['nao']} | {row['initial_energy_hartree']:.9f} | "
                      f"{row['last_energy_hartree']:.9f} | {row['last_decrease_millihartree']:.4f} |" for row in metrics)
    hashes = '\n'.join(f"- `{key}`：`{value}`" for key, value in identity.items() if key.endswith('sha256'))
    software = ', '.join(f'{name} {value}' for name, value in metadata['versions'].items())
    terminal = (f"训练进程记录耗时：{summary['elapsed_seconds']:.2f} s。" if summary else '尚无训练终态 summary，不能据快照宣称任务完成。')
    timing = (f"已记录的 {len(times)} 个更新 epoch 中，耗时中位数为 {float(np.median(times)):.3f} s"
              if times else '尚未取得可解析的逐 epoch 耗时日志')
    failure = f"\n失败信息：`{summary['error']}`。最后有效检查点保留于 epoch {last['epoch']}。\n" if status == 'failed' else ''
    report = f"""# GMTKN55 首批共享 NNAO 训练结果

快照生成时间：{datetime.now(timezone.utc).isoformat()}；状态：**{status_text}**。
请求 **{metadata['target_epochs']} epochs**，当前已接受 **{last['epoch']} epochs**，包含 epoch 0 和 {len(last['history']) - 1} 次完整 batch Adam 更新。

## 实测结果

24 条训练结构的等权平均能量从 **{initial['loss_hartree']:.12f} Ha** 到 **{last['loss_hartree']:.12f} Ha**，下降 **{decrease:.6f} mHa**。
最后有效参数：epoch **{last['epoch']}**；最优平均能量参数：epoch **{best['epoch']}**，loss **{best['loss_hartree']:.12f} Ha**。
两者分别来自 `last.npz` 与 `best.npz`，能量和逐分子诊断来自各自参数的实际 SCF 评估；best 表示全体均值最小的共享参数，不表示每个分子独立的能量最优值。
{sum(value > 0 for value in deltas)}/{len(deltas)} 个结构相对初始点降能；逐结构变化为 {min(deltas):.6f} 至 {max(deltas):.6f} mHa。均值目标允许不同结构间发生权衡。
{terminal} {timing}（包含评估、优化与检查点写入；不包含 epoch 0 的初始编译评估）。
{failure}
![平均降能与逐体系降能](training_curve.png)

| 体系 | AO 数 | 初始能量 / Ha | 当前有效能量 / Ha | 降能 / mHa |
|---|---:|---:|---:|---:|
{table}

## 方法、目标与作用范围

本轮是 **一套共享 MACE 参数** 的训练，共 {identity['parameter_count']} 个参数；全部 24 条 GMTKN55 v2 中性、spin=0、CHNOF 结构都参与训练，**没有 held-out 集**。
loss 为 `mean(E_RHF,DF)`，单位 Hartree，全部结构等权。GMTKN55 参考能量、反应标签、拟合误差不进入 loss；没有额外正则项。
基组为球谐 `szp663_direct/core6`、全电子无 ECP。固定几何、primitive 指数与内层；MACE 学习价层及多 primitive 极化壳的收缩系数，包括 H 的 p 壳和重原子的 d 壳。
MACE：channels={identity['model_config']['channels']}、interaction={identity['model_config']['num_interactions']}、max_ell={identity['model_config']['max_ell']}、correlation={identity['model_config']['correlation']}、zero_init=True、seed={identity['seed']}。
每个 epoch 在同一共享参数处重收敛全部 24 次 SCF，平均梯度后执行一次 `{identity['optimizer']}`，学习率 `{identity['learning_rate']}`。
SCF：max_cycle={identity['scf_config']['max_cycle']}、能量容差 {identity['scf_config']['conv_tol']} Ha、密度容差 {identity['scf_config']['conv_tol_density']}、轨道梯度容差 {identity['scf_config']['conv_tol_grad']}。
backward 是收敛密度固定点 implicit SCF，线性求解 tolerance={identity['implicit_tolerance']}、max_iter={identity['implicit_config']['max_iter']}、restart={identity['implicit_config']['restart']}。
独立模型的 24 体系 FD/5 步 Adam 是训练前验证阶段；本报告只统计之后的共享模型训练阶段，二者结果不混合。
{validation_text}

## 硬件、后端与复现记录

`jy_gpu_2` / 节点 `{metadata['hostname']}` / Slurm `{metadata['slurm_job_id']}`；GPU `{', '.join(metadata['device_kinds'])}`；JAX backend `{metadata['backend']}`，设备 `{', '.join(metadata['devices'])}`。
申请单 GPU、16 CPU、64 GiB、4 小时；固定 primitive S/H 与压缩 DF factors 来自 native CPU 缓存，辅助基 `{identity['auxbasis']}`，GPU 执行收缩、SCF、MACE 与反传。没有保存完整四指标 ERI。
参数向量 dtype `{metadata['parameter_dtype']}`，RHF 积分/计算 dtype `{identity['dtype']}`；Python {metadata['python_version']}；{software}。
运行命令与分配见 `../run_train.sbatch`；完整配置、源码导入位置与每个 DF cache SHA-256 见 `../{args.training_dir.name}/metadata.json`，逐分子数值见 `metrics.csv`。

## 检查与证据边界

每个接受步骤检查 SCF converged、轨道残差 ≤1e-7、最小 overlap 特征值 ≥1e-8、能量重构误差 ≤1e-9 Ha，且能量和梯度有限；失败即停止，不隐藏剔除结构。
当前检查点的最大轨道残差为 {max(row['final_orbital_residual'] for row in metrics):.3e}，最大重构误差为 {max(row['final_reconstruction_error'] for row in metrics):.3e} Ha。
本结果证明已选训练样本上的变分能量优化；不据此宣称泛化性能、double/triple-zeta 水平、轨道稳定性或全局最低 HF 解。DF 近似与固定 primitive 池也是适用范围的一部分。
`last.npz` 与 `best.npz` 保留参数、完整 Adam state/counter、梯度和其对应的完整接受历史；恢复要求数据、源码、模型、缓存和优化配置一致。

## 数据与源码 SHA-256

{hashes}

最后参数数组 SHA-256：`{last['parameter_sha256']}`；best 参数数组 SHA-256：`{best['parameter_sha256']}`。
"""
    (args.output_dir / 'README.md').write_text(report)
    verification = dict(status=status, accepted_epoch=last['epoch'], target_epochs=metadata['target_epochs'],
        mean_initial_energy_hartree=initial['loss_hartree'], mean_last_energy_hartree=last['loss_hartree'],
        mean_decrease_millihartree=decrease, best_epoch=best['epoch'], best_loss_hartree=best['loss_hartree'],
        consistent_checkpoint_loss=True, structures=len(metrics),
        files={str(path): sha256(path) for path in (args.training_dir / 'metadata.json', args.training_dir / 'last.npz', args.training_dir / 'best.npz')})
    if args.validation_summary.exists():
        verification['validation_summary_sha256'] = sha256(args.validation_summary)
        if verification['validation_summary_sha256'] != identity['validation_summary_sha256']:
            raise ValueError('Training used a different validation summary.')
    (args.output_dir / 'verification.json').write_text(json.dumps(verification, indent=2, sort_keys=True) + '\n')
    print(json.dumps(verification, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
