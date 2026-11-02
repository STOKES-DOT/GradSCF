"""Combine the added fixed-SZP controls with immutable ethane references."""
from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'nnao-ethane-torsion-20261018'
P2 = OUT.parent/'nnao-ethane-p2-20261018'
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

FACTOR = 627.5094740631
CUSTOM = 'STO-6G + C(d), H(p)'
MACE = 'NNAO / MACE'
FREE = 'NNAO / free contraction'
EXTRA_P = 'NNAO / C p2 / free'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    report = json.loads((OUT/'results.json').read_text())
    assert report['status'] == 'complete' and report['passed'] == 75
    assert report['geometry_sha256'] == sha(OLD/'structures-dense.json')
    assert report['basis_library_sha256'] == sha(OUT/'basis-library.json')
    assert all(r['status'] == 'passed' and r['orbital_residual'] < 1e-7 for r in report['results'])
    baseline = list(csv.DictReader((OLD/'benchmark-curves.csv').open()))
    p2 = [r for r in csv.DictReader((P2/'curves.csv').open()) if r['basis'] == EXTRA_P]
    assert len(baseline) == 175 and len(p2) == 25
    rows = baseline+p2+report['results']
    angles = np.arange(0, 121, 5)
    curves = {}
    for name in dict.fromkeys(r['basis'] for r in rows):
        selected = sorted([r for r in rows if r['basis'] == name], key=lambda r: float(r['angle_deg']))
        np.testing.assert_array_equal([float(r['angle_deg']) for r in selected], angles)
        assert len(set(int(r['nao']) for r in selected)) == 1
        energy = np.array([float(r['energy_hartree']) for r in selected])
        assert np.isfinite(energy).all()
        curves[name] = dict(energy=energy, relative=(energy-energy[0])*FACTOR,
                            nao=int(selected[0]['nao']))
    assert len(curves) == 11
    reference = curves['cc-pVTZ']['relative']
    summary, long = [], []
    for name, c in curves.items():
        diff = c['relative']-reference
        summary.append(dict(basis=name, nao=c['nao'], zero_energy_hartree=float(c['energy'][0]),
            grid_barrier_kcal_mol=float(np.ptp(c['relative'])),
            barrier_difference_vs_ccpvtz_kcal_mol=float(np.ptp(c['relative'])-np.ptp(reference)),
            max_curve_difference_vs_ccpvtz_kcal_mol=float(abs(diff).max()),
            rms_curve_difference_vs_ccpvtz_kcal_mol=float(np.sqrt(np.mean(diff**2))),
            closure_error_hartree=float(abs(c['energy'][-1]-c['energy'][0])),
            mirror_max_error_hartree=float(abs(c['energy']-c['energy'][::-1]).max())))
        for angle, energy, relative in zip(angles, c['energy'], c['relative']):
            long.append(dict(basis=name, angle_deg=int(angle), energy_hartree=float(energy),
                             relative_energy_kcal_mol=float(relative), nao=c['nao']))
    write_csv(OUT/'comparison-summary.csv', summary)
    write_csv(OUT/'all-curves.csv', long)
    colors = {'STO-3G': '#B0B4B8', '3-21G': '#77838C', 'STO-6G': '#99A2AA',
              'STO-6G + C(d)': '#9F9259', CUSTOM: '#8753A0', '6-31G(d,p)': '#7C8938',
              'cc-pVDZ': '#B95F83', 'cc-pVTZ': '#26313A', MACE: '#D87632',
              FREE: '#28699C', EXTRA_P: '#258580'}
    styles = {CUSTOM: ('-', 'D'), MACE: ('-', 'o'), FREE: ('--', 's'),
              EXTRA_P: ('-.', '^'), '6-31G(d,p)': ('-.', None),
              'cc-pVDZ': ('--', None), 'cc-pVTZ': ('-', None)}
    labels = {CUSTOM: 'STO-6G + C-d / H-p (44 AO)', MACE: 'NNAO / MACE (44 AO)',
              FREE: 'NNAO / free (44 AO)', EXTRA_P: 'NNAO / free + C-p (50 AO)',
              '6-31G(d,p)': '6-31G(d,p) (58 AO)', 'cc-pVDZ': 'cc-pVDZ (58 AO)',
              'cc-pVTZ': 'cc-pVTZ (144 AO)'}
    focus = [CUSTOM, MACE, FREE, EXTRA_P, '6-31G(d,p)', 'cc-pVDZ', 'cc-pVTZ']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#26313A', 'text.color': '#26313A'})
    fig, axes = plt.subplots(2, 1, figsize=(10.3, 8.6), sharex=True,
                              gridspec_kw={'height_ratios': [1.6, 1], 'hspace': .12})
    for name in focus:
        c = curves[name]
        linestyle, marker = styles[name]
        for ax, values in [(axes[0], c['relative']), (axes[1], c['relative']-reference)]:
            ax.plot(angles, values, color=colors[name], ls=linestyle, marker=marker,
                    markevery=2, markersize=4, lw=2 if name in [CUSTOM, MACE, FREE, 'cc-pVTZ'] else 1.5,
                    label=labels[name])
    axes[0].set_ylabel(r'$E(\phi)-E(0)$ (kcal/mol)')
    axes[0].legend(frameon=False, ncol=2, fontsize=9.2, loc='lower center', bbox_to_anchor=(.5, 1.02))
    axes[1].set_ylabel('Difference from HF/cc-pVTZ\nrelative curve (kcal/mol)')
    axes[1].axhline(0, color='#59636B', lw=.8)
    axes[1].set_xlabel('Methyl rotation from the initial staggered geometry (degrees)')
    for ax in axes:
        ax.grid(color='#E5E8EA', lw=.65)
        ax.set_xlim(-2, 122)
        ax.set_xticks(np.arange(0, 121, 15))
        ax.axvline(60, color='#A9B0B5', ls=':', lw=.8)
    fig.suptitle('Ethane torsion: an explicit fixed-SZP control at 44 AOs', fontsize=14, y=.985)
    fig.text(.12, .025, '25 rigid geometries; DF-RHF / def2-universal-jkfit, spherical AOs. NNAO MACE update 3351.\n'
        'Fixed control: unchanged STO-6G + C-d exponent 0.8 + H-p exponent 1.1 (Bohr^-2).\n'
        'Each relative curve uses its own zero. HF/cc-pVTZ is a comparison, not an exact reference.',
        fontsize=8.5, color='#59636B')
    fig.subplots_adjust(left=.12, right=.98, top=.76, bottom=.15)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'ethane-relative-comparison.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    fig = plt.figure(figsize=(12.5, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], hspace=.35, wspace=.37)
    overview = fig.add_subplot(grid[:, 0])
    same_ao = fig.add_subplot(grid[0, 1])
    nnao = fig.add_subplot(grid[1, 1])
    for ax, selected, title in [(overview, list(curves), 'All 11 basis curves'),
                               (same_ao, [CUSTOM, MACE, FREE], 'Same 44-AO dimension'),
                               (nnao, [MACE, FREE, EXTRA_P], 'NNAO absolute energies')]:
        for name in selected:
            linestyle, marker = styles.get(name, (':', None))
            ax.plot(angles, curves[name]['energy'], color=colors[name], ls=linestyle,
                    marker=marker, markevery=3, markersize=3.5,
                    lw=2 if name in [CUSTOM, MACE, FREE] else 1.2,
                    label=labels.get(name, f"{name} ({curves[name]['nao']} AO)"))
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel('Methyl rotation (degrees)')
        ax.set_ylabel('SCF total energy (Eh)')
        ax.set_xlim(-3, 123)
        ax.set_xticks(np.arange(0, 121, 30))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f' if ax is nnao else '%.3f'))
        ax.grid(color='#E5E8EA', lw=.65)
    handles, legend_labels = overview.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, .93), fontsize=8.5)
    fig.suptitle('Ethane torsion: absolute total energies', fontsize=14, y=.99)
    fig.text(.08, .025, 'Original total energies in Hartree; no per-curve energy shift. Same geometry, DF approximation and electronic method.\n'
             'Equal AO count does not equalize primitive pools, core contractions, or environment adaptation.',
             fontsize=8.5, color='#59636B')
    fig.subplots_adjust(left=.08, right=.98, top=.73, bottom=.13)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'ethane-absolute-comparison.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    assert max(r['closure_error_hartree'] for r in summary) < 1e-8
    assert max(r['mirror_max_error_hartree'] for r in summary) < 1e-8
    data = dict(method='DF-RHF', auxbasis='def2-universal-jkfit', angles_deg=angles.tolist(),
        new_job_id=report['job_id'], new_scf_count=75, new_elapsed_seconds=report['elapsed_seconds'],
        new_backend='GPU4PySCF', new_device=report['device'], new_cpus=report['cpus'],
        max_new_orbital_residual=max(r['orbital_residual'] for r in report['results']),
        reused_mace_checkpoint_update=3351, results=summary,
        inputs_sha256={str(path): sha(path) for path in [OUT/'results.json', OUT/'basis-library.json',
            OLD/'benchmark-curves.csv', P2/'curves.csv', OLD/'structures-dense.json']})
    (OUT/'comparison-summary.json').write_text(json.dumps(data, indent=2)+'\n')
    lines = ['# 乙烷扭转：STO-6G 固定极化基组与 NNAO 对照', '',
        '在原 25 个刚性构型（0–120°，间隔 5°）上，新增 STO-6G、STO-6G+C-d、'
        'STO-6G+C-d+H-p 三套固定基组，分别为 16、26、44 AO。C-d 指数 0.8，H-p 指数 1.1 Bohr^-2，'
        '均复制自原 benchmark 的 6-31G(d,p) 极化壳层。STO-6G 部分原样使用 PySCF 内置数据。', '',
        '这是明确指定参数的自定义对照，不宣称存在唯一标准化 STO-6G* 或 STO-6G**。', '',
        '| 基组 | AO | 0° 总能量 / Eh | 扭转势垒 / kcal mol-1 | 相对曲线最大差 / kcal mol-1 | RMS差 / kcal mol-1 |',
        '|---|---:|---:|---:|---:|---:|']
    order = ['STO-6G', 'STO-6G + C(d)', CUSTOM, MACE, FREE, EXTRA_P,
             '6-31G(d,p)', 'cc-pVDZ', 'cc-pVTZ', 'STO-3G', '3-21G']
    for name in order:
        row = next(r for r in summary if r['basis'] == name)
        lines.append(f"| {name} | {row['nao']} | {row['zero_energy_hartree']:.10f} | "
                     f"{row['grid_barrier_kcal_mol']:.6f} | {row['max_curve_difference_vs_ccpvtz_kcal_mol']:.6f} | "
                     f"{row['rms_curve_difference_vs_ccpvtz_kcal_mol']:.6f} |")
    lines += ['', '差值以 HF/cc-pVTZ 为比较；它不是精确极限。势垒来自固定几何的刚性扫描。', '',
        '本例中固定 44 AO 对照的相对扭转曲线优于当前 44 AO NNAO 的 MACE 和自由收缩端点，'
        '但 NNAO 的绝对总能量更低。总能量下降不保证相对能量误差同步减小。'
        '不同基组的误差抵消需要在其他任务上另外验证；本例不能推导一种基组全面优于另一种。', '',
        'NNAO 曲线沿用 update 3351，不是后来的 epoch 596 检查点。50 AO 曲线仅是新增第二套 C-p 的自由收缩结果。'
        '相同 AO 数仅匹配矩阵维度，primitive 池、内层和归一化后径向函数均不同；'
        '总能量差不能全部归于 MACE 的环境自适应。', '',
        f"新增计算：GPU4PySCF，任务 {report['job_id']}，{report['device']}，32 CPU；75/75 SCF通过，"
        f"数值循环耗时 {report['elapsed_seconds']:.2f} 秒（不含 Python 导入和 Slurm 排队）。"
        f"最大轨道残差 {data['max_new_orbital_residual']:.3g}。", '',
        '计算为 DF-RHF/def2-universal-jkfit，float64，spherical AO；能量收敛阈值 1e-12 Eh，'
        '轨道梯度阈值 1e-9，最终残差 <1e-7。未新做有限差分或完整稳定性扫描。', '',
        '文件：`ethane-relative-comparison.png/.pdf`、`ethane-absolute-comparison.png/.pdf`，'
        '`all-curves.csv`（11×25 点）、`comparison-summary.csv/.json`、`basis-library.json`。', '',
        '计算命令：`sbatch artifacts/nnao-ethane-sto6g-polarized-20261020/run.slurm`。',
        '绘图命令：`/opt/anaconda3/bin/python artifacts/nnao-ethane-sto6g-polarized-20261020/analyze.py`。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
