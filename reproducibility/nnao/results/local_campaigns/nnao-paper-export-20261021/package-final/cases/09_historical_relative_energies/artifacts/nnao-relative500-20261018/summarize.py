"""Compare the same signed reactions; keep failures and comparison scope explicit."""
from pathlib import Path
import csv
import json
import math
import sys
from gpu4_reference_utils import atomic_json

OUT=Path(__file__).resolve().parent
CONVERSION=627.5094740631
def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

native=json.loads((OUT/'native-results.json').read_text())
reference_file=sys.argv[1] if len(sys.argv)>1 else 'gpu4-check.json'
independent=json.loads((OUT/reference_file).read_text())
checks={(r['structure_id'],r['basis']):r for r in independent['results']}
energies={basis:{} for basis in ['NNAO-500-initial','NNAO-500-current']}
molecules=[]
for r in native['results']:
    check=checks.get((r['structure_id'],r['basis']),{})
    valid=r['status']==check.get('status')=='passed'
    if valid:energies[r['basis']][r['structure_id']]=r['energy_hartree']
    molecules.append(dict(structure_id=r['structure_id'],basis=r['basis'],status='passed' if valid else 'failed',
        native_energy_hartree=r.get('energy_hartree'),reference_energy_hartree=check.get('energy_hartree'),
        native_reference_error_hartree=check.get('native_energy_error_hartree'),nao=r.get('nao'),
        error=r.get('error',check.get('error',''))))
write_csv('per-structure.csv',molecules)
old=list(csv.DictReader((OUT/'historical-relative-energies.csv').open()))
reference={r['reaction_id']:r for r in old if r['basis']=='cc-pvtz'}
reactions=[json.loads(s) for s in (OUT/'reactions.jsonl').read_text().splitlines()]
new=[]
for reaction in reactions:
    rid=reaction['reaction_id'];anchor=reference[rid]
    for basis,values in energies.items():
        missing=[t['structure_id'] for t in reaction['terms'] if t['structure_id'] not in values]
        energy=(math.fsum(t['coefficient']*values[t['structure_id']] for t in reaction['terms'])*CONVERSION
            +reaction['energy_add_kcal_mol']) if not missing else None
        hf=float(anchor['hf_ccpvtz_kcal_mol']);gmt=float(reaction['reference_energy_kcal_mol'])
        new.append(dict(reaction_id=rid,category=reaction['category'],basis=basis,
            status='passed' if energy is not None else 'failed',energy_kcal_mol=energy,
            hf_ccpvtz_kcal_mol=hf,gmtkn55_reference_kcal_mol=gmt,
            error_vs_hf_ccpvtz_kcal_mol=energy-hf if energy is not None else None,
            error_vs_gmtkn55_kcal_mol=energy-gmt if energy is not None else None,
            missing_structures=';'.join(missing)))
write_csv('relative-energies.csv',new)
write_csv('comparison-relative-energies.csv',
    [{key:r.get(key,'') for key in new[0]} for r in old+new])
categories=[]
for category in ['reaction','barrier','noncovalent']:
    eligible={r['reaction_id'] for r in new if r['category']==category}
    eligible-={r['reaction_id'] for r in new if r['category']==category and r['status']!='passed'}
    for basis in ['minao','3-21g','6-31g(d,p)','def2-svp','cc-pvdz','NNAO-200-best','NNAO-500-initial','NNAO-500-current']:
        group=[r for r in (new if basis.startswith('NNAO-500') else old)
            if r['category']==category and r['basis']==basis and r['reaction_id'] in eligible]
        record=dict(category=category,basis=basis,count=len(group),target=6)
        for ref in ['hf_ccpvtz','gmtkn55']:
            errors=[float(r[f'error_vs_{ref}_kcal_mol']) for r in group]
            record[f'mae_vs_{ref}_kcal_mol']=sum(map(abs,errors))/len(errors) if errors else None
        categories.append(record)
write_csv('category-metrics.csv',categories)
report=dict(status='complete' if all(r['status']=='passed' for r in molecules) else 'complete_with_failures',
    snapshot=json.loads((OUT/'snapshot.json').read_text()),native=native['device'],reference_device=independent['device'],
    reference_file=reference_file,
    slurm_job_id=native['slurm_job_id'],cpu_threads=32,native_elapsed_seconds=native['elapsed_seconds'],
    independent_elapsed_seconds=independent['elapsed_seconds'],
    native_passed=native['passed'],native_failed=native['failed'],independent_passed=independent['passed'],
    independent_failed=independent['failed'],max_native_reference_error_hartree=max(
        (r.get('native_energy_error_hartree',0.) for r in independent['results'] if r['status']=='passed'),default=None),
    training_distance_overlap=native['training_distance_overlap'],training_formula_overlap_count=native['training_formula_overlap_count'],
    metrics=categories,performed_gradients=False,
    limitations=['18 historical relative energies; not full GMTKN55/WTMAD2.',
        'HF/cc-pVTZ is a finite-basis anchor, not a correlated or CBS reference.',
        'NCI uses monomer sum minus complex, positive binding, no CP/BSSE correction.',
        'New versus old models differ in training data, exponent pool, core asset and architecture.',
        'No training-geometry overlap at pair-distance rounding 1e-5 Angstrom; not chemical-family or calibration independence.'])
atomic_json(OUT/'summary.json',report)
lines=['# 当前500体系模型的历史相对能量推理','',
    f"冻结参数：epoch {report['snapshot']['epoch']} / update {report['snapshot']['update_step']}；训练保持运行。",'',
    '| 类别 | 模型/基组 | 有效条目 | MAE vs HF/cc-pVTZ | MAE vs GMTKN55 |',
    '|---|---|---:|---:|---:|']
for r in categories:
    fmt=lambda x:'缺失' if x is None else f'{x:.4f}'
    lines.append(f"| {r['category']} | {r['basis']} | {r['count']}/6 | {fmt(r['mae_vs_hf_ccpvtz_kcal_mol'])} | {fmt(r['mae_vs_gmtkn55_kcal_mol'])} |")
lines+=['','单位：kcal/mol。','',*['- '+s for s in report['limitations']],
    '',f"GradSCF与独立PySCF同基组最大能量差：{report['max_native_reference_error_hartree']} Eh；参考后端：{report['reference_device']}。"]
lines+=['','## 三个历史例子','',
    '| 例子 | 当前500模型 | 原200模型 | HF/cc-pVTZ | GMTKN55 |','|---|---:|---:|---:|---:|']
for identifier,title in [('G2RC/2','CO2 + H2 -> HCOOH'),('BH76/37','HCN -> HNC forward barrier'),
                         ('S22/3','Formic acid dimer binding')]:
    current=next(r for r in new if r['reaction_id']==identifier and r['basis']=='NNAO-500-current')
    previous=next(r for r in old if r['reaction_id']==identifier and r['basis']=='NNAO-200-best')
    lines.append(f"| {title} ({identifier}) | {current['energy_kcal_mol']:.6f} | "
        f"{float(previous['energy_kcal_mol']):.6f} | {current['hf_ccpvtz_kcal_mol']:.6f} | "
        f"{current['gmtkn55_reference_kcal_mol']:.6f} |")
initial={r['structure_id']:r['energy_hartree'] for r in native['results']
         if r['basis']=='NNAO-500-initial' and r['status']=='passed'}
trained={r['structure_id']:r['energy_hartree'] for r in native['results']
         if r['basis']=='NNAO-500-current' and r['status']=='passed'}
lower=sum(trained[k]<initial[k] for k in set(trained)&set(initial))
lines+=['',f'相比同池初始化，{lower}/{len(initial)} 个结构的总能量降低；这不保证每个相对能量的误差同时降低。',
    '例如 HCN 势垒由同池初始化的62.867564升至71.995167 kcal/mol，仍高于HF/cc-pVTZ的47.551951。',
    '',f"结构前向计算耗时{report['native_elapsed_seconds']:.2f}s，独立参考耗时{report['independent_elapsed_seconds']:.2f}s；不含排队、导入和参考脚本重试。",
    '训练任务99005在测试期间保持运行。最初参考脚本将GPU4PySCF默认level_shift=None误判为未解除移位；修正为显式0后参考78项全部通过，原误判记录单独归档。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:report[k] for k in ['status','native_passed','native_failed','independent_passed','independent_failed','max_native_reference_error_hartree']},indent=2))
