"""Source-backed energy/AO comparisons for the frozen epoch-594 benchmark."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'gpu4/comparison.json').read_text())
native=json.loads((ROOT/'native-results.json').read_text())
snapshot=json.loads((ROOT/'snapshot.json').read_text())
assert report['status']=='passed' and report['passed']==150 and report['failed']==0
assert native['status']=='complete' and native['passed']==20
structures=[json.loads(s) for s in (ROOT/'test-selection.jsonl').read_text().splitlines()]
identifiers=[r['structure_id'] for r in structures]
rows=report['results'];index={(r['structure_id'],r['basis']):r for r in rows}
assert len(index)==150
bases=list(dict.fromkeys(r['basis'] for r in rows))
current='NNAO-200-epoch594';previous='NNAO-pilot-best975';initial='NNAO-200-initial'
summary=[]
for basis in bases:
    cases=[index[i,basis] for i in identifiers]
    summary.append(dict(basis=basis,cases=len(cases),mean_ao=float(np.mean([r['nao'] for r in cases])),
        mean_gap_mEh=float(np.mean([1000*(index[i,basis]['energy_hartree']-index[i,'cc-pvtz']['energy_hartree']) for i in identifiers])),
        current_lower_count=sum(index[i,current]['energy_hartree']<index[i,basis]['energy_hartree']-1e-8 for i in identifiers)))
comparisons=[]
for i in identifiers:
    energy=lambda b:index[i,b]['energy_hartree']
    comparisons.append(dict(structure_id=i,nao=index[i,current]['nao'],fock_dimension=index[i,current]['nao'],
        current_energy_Eh=energy(current),old24_energy_Eh=energy(previous),
        current_gap_mEh=1000*(energy(current)-energy('cc-pvtz')),
        old24_gap_mEh=1000*(energy(previous)-energy('cc-pvtz')),
        improvement_over_old24_mEh=1000*(energy(previous)-energy(current)),
        improvement_over_current_initial_mEh=1000*(energy(initial)-energy(current)),
        nao_631g=index[i,'6-31g']['nao'],nao_ccpvdz=index[i,'cc-pvdz']['nao'],nao_ccpvtz=index[i,'cc-pvtz']['nao']))
def csv_file(path,values):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]))
        writer.writeheader();writer.writerows(values)
csv_file(ROOT/'per-molecule.csv',comparisons)
csv_file(ROOT/'per-basis.csv',summary)
csv_file(ROOT/'all-results.csv',[dict(structure_id=r['structure_id'],basis=r['basis'],energy_Eh=r['energy_hartree'],
    gap_ccpvtz_mEh=r['above_cc_pvtz_millihartree'],nao=r['nao'],primitive_nao=r['primitive_nao'],
    fock_dimension=r['fock_dimension'],cold_scf_seconds=r['timings'][0]['scf_seconds'],
    warm_scf_seconds=r['timings'][1]['scf_seconds'],native_error_Eh=r.get('native_energy_error_hartree','')) for r in rows])
parity=max(r.get('native_energy_error_hartree',0.) for r in rows if r['basis']==current)
improvement=float(np.mean([r['improvement_over_old24_mEh'] for r in comparisons]))
template_improvement=float(np.mean([r['improvement_over_current_initial_mEh'] for r in comparisons]))
result=dict(snapshot=snapshot,structures=10,valid_cases=150,native_cases=20,
    mean_improvement_over_old24_mEh=improvement,mean_improvement_over_current_initial_mEh=template_improvement,
    current_native_gpu4_max_error_Eh=parity,per_basis=summary,per_molecule=comparisons,
    gpu=report['provenance']['gpu_name'],job_id=report['slurm_job_id'])
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[1,1.25]})
names=[r['basis'] for r in summary];values=[r['mean_gap_mEh'] for r in summary]
colors=['#196b91' if n==current else '#b77935' if n==previous else '#c8cdd2' for n in names]
axes[0].barh(np.arange(len(names)),values,color=colors)
axes[0].set_yticks(np.arange(len(names)),names);axes[0].invert_yaxis()
axes[0].set_xscale('symlog',linthresh=10)
axes[0].set_xlabel('Mean E − E(cc-pVTZ), mEh / molecule\nSymmetric log scale; linear within ±10 mEh')
axes[0].set_title('All basis sets: finite-basis energy gap',loc='left')
axes[0].grid(axis='x',alpha=.18)
delta=np.array([r['improvement_over_old24_mEh'] for r in comparisons])
axes[1].barh(np.arange(10),delta,color=['#196b91' if x>=0 else '#7b838a' for x in delta])
axes[1].set_yticks(np.arange(10),identifiers);axes[1].invert_yaxis()
axes[1].axvline(0,color='#444444',linewidth=.8)
axes[1].set_xlabel('E(old 24-system model) − E(current 200-system model)\nmEh / molecule; positive = current model lower')
axes[1].set_title('Current epoch 594 versus old epoch 975',loc='left')
axes[1].grid(axis='x',alpha=.18)
fig.suptitle('NNAO frozen-basis benchmark | 10 historical independent structures',fontsize=15)
fig.text(.5,.015,'All-electron DF-RHF | spherical AO | def2-universal-jkfit | cc-pVTZ is a finite-basis reference, not CBS',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.95))
for ext in ['png','pdf']:fig.savefig(ROOT/f'comparison.{ext}',dpi=200)

lines=['# 当前200体系模型与历史基组benchmark对比','',
    f"冻结任务96984的最佳检查点：epoch {snapshot['checkpoint_epoch']}，SHA256 `{snapshot['checkpoint_sha256']}`。测试期间没有拟合、梯度更新或优化器更新。",'',
    '沿用历史10个测试结构，重新计算12个传统基组、旧24体系best975、当前200体系epoch594，以及当前指数池的初始收缩。共150项GPU4PySCF计算和20项当前NNAO的GradSCF原生计算。',
    '测试结构ID及整体分子式与当前200训练结构无交集；这不保证片段或反应关联独立。','',
    '统一全电子RHF、球谐AO、float64、def2-universal-jkfit；hcore初猜，conv_tol=1e-12 Eh、conv_tol_grad=1e-9、max_cycle=200。失败后允许0.2 level shift，最终须在原始Fock上收敛。当前NNAO原生路径还检查内部稳定性；同基组GPU4PySCF能量必须在1e-7 Eh内一致。','',
    f"全部150项比较有效。当前NNAO原生/GPU4PySCF最大能量差为 {parity:.3e} Eh。硬件：{result['gpu']}，Slurm作业 {result['job_id']}，32 CPU。",'',
    f'当前模型相对旧24体系模型平均降低 {improvement:.3f} mEh/体系；相对当前指数池初始收缩平均降低 {template_improvement:.3f} mEh/体系。正值表示当前模型能量更低。','',
    '| 体系 | NNAO AO / Fock尺寸 | 旧24 − cc-pVTZ / mEh | 当前200 − cc-pVTZ / mEh | 当前相对旧24降低 / mEh |',
    '|---|---:|---:|---:|---:|']
for r in comparisons:
    n=r['nao'];lines.append(f"| {r['structure_id']} | {n} / {n}×{n} | {r['old24_gap_mEh']:.3f} | {r['current_gap_mEh']:.3f} | {r['improvement_over_old24_mEh']:.3f} |")
lines+=['','| 基组 | 平均AO数 | 平均相对cc-pVTZ能量差 / mEh | 当前模型能量更低的体系数 |','|---|---:|---:|---:|']
for r in summary:
    score='—' if r['basis']==current else f"{r['current_lower_count']}/10"
    lines.append(f"| {r['basis']} | {r['mean_ao']:.1f} | {r['mean_gap_mEh']:.3f} | {score} |")
lines+=['','当前和旧模型的训练轮数、训练数据及指数池均不同，本比较评价完整基组表现，不能把差异单独归因于训练样本数量。cc-pVTZ仅为同几何有限基组DF-HF参照，不是CBS；结果不能直接推广到力、BSSE或完整GMTKN55。',
    '计时数据在all-results.csv中；JAX首次调用含编译，GPU4PySCF首次kernel含DF准备，两个后端计时范围不同，不计算速度比。','','![基组比较](comparison.png)']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['snapshot','per_basis','per_molecule']}))
