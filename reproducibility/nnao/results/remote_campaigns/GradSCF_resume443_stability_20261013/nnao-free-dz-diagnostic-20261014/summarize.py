"""Same-pool free SZ/DZ and conventional-basis comparison for three reactions."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/'nnao-relative200-20261014'
SZOUT=OUT.parent/'nnao-free-sz-diagnostic-20261014'
FACTOR=627.5094740631

def main():
    report=json.loads((OUT/'summary.json').read_text())
    reactions=json.loads((OUT/'reactions.json').read_text())
    dz={r['structure_id']:r for r in report['results']}
    sz={r['structure_id']:r for r in json.loads((SZOUT/'summary.json').read_text())['results']}
    original=json.loads((PARENT/'gpu4/comparison.json').read_text())['results']
    baselines={b:{r['structure_id']:r['energy_hartree'] for r in original
                  if r['basis']==b and r['status']=='passed'}
               for b in ['NNAO-200-best','minao','3-21g','6-31g(d,p)','def2-svp','cc-pvdz','cc-pvtz']}
    baselines['free-SZ']={i:r['best_energy_hartree'] for i,r in sz.items() if r['status'] in ['complete','partial']}
    baselines['free-DZ']={i:r['best_energy_hartree'] for i,r in dz.items() if r['status'] in ['complete','partial']}
    rows=[]
    for reaction in reactions:
        ids={t['structure_id'] for t in reaction['terms']}
        def relative(e):
            return math.fsum(t['coefficient']*e[t['structure_id']] for t in reaction['terms'])*FACTOR+reaction['energy_add_kcal_mol']
        anchor=relative(baselines['cc-pvtz'])
        for label,energies in baselines.items():
            value=relative(energies) if ids<=energies.keys() else None
            rows.append(dict(reaction_id=reaction['reaction_id'],category=reaction['category'],basis=label,
                status='available' if value is not None else 'missing',energy_kcal_mol=value,
                hf_ccpvtz_kcal_mol=anchor,gmtkn55_kcal_mol=reaction['reference_energy_kcal_mol'],
                error_vs_hf_kcal_mol=None if value is None else value-anchor,
                error_vs_gmtkn55_kcal_mol=None if value is None else value-reaction['reference_energy_kcal_mol']))
    with (OUT/'relative-energies.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ranges=[]
    for reaction in reactions:
        low=high=0.;complete=True
        for term in reaction['terms']:
            endpoints=[s['energy_hartree'] for s in dz[term['structure_id']]['starts']
                       if 'energy_hartree' in s]
            if not endpoints:complete=False;break
            c=term['coefficient'];a,b=min(endpoints),max(endpoints)
            low+=c*(a if c>=0 else b);high+=c*(b if c>=0 else a)
        ranges.append(dict(reaction_id=reaction['reaction_id'],
            endpoint_combination_min_kcal_mol=low*FACTOR+reaction['energy_add_kcal_mol'] if complete else None,
            endpoint_combination_max_kcal_mol=high*FACTOR+reaction['energy_add_kcal_mol'] if complete else None,
            note='Sensitivity envelope over finite-budget start endpoints; not a confidence interval or rigorous optimum bound.'))
    with (OUT/'endpoint-combination-ranges.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ranges[0]));writer.writeheader();writer.writerows(ranges)
    count=Counter(s['status'] for r in dz.values() for s in r['starts'])
    lines=['# 同池自由 SZ 与 DZ 对照','',
        '只增加价层第二个独立径向通道。固定原始 primitive 池、内层、极化壳层数量、几何、电子态及辅助基。',
        '每结构 4 起点、每起点最多 500 步，使用与 SZ 一致的 Adam、收敛检查与隐式梯度。',
        '当前最佳 SZ 被嵌入 DZ 初始空间；DZ 的起点条件不同，有限预算结果不等于两个空间的全局极限。','',
        '| 条目 | 基组 | 相对能量 | 偏差 vs HF/cc-pVTZ | 偏差 vs GMTKN55 |',
        '|---|---|---:|---:|---:|']
    fmt=lambda x:'缺失' if x is None else f'{x:.6f}'
    for r in rows:
        lines.append(f"| {r['reaction_id']} | {r['basis']} | {fmt(r['energy_kcal_mol'])} | {fmt(r['error_vs_hf_kcal_mol'])} | {fmt(r['error_vs_gmtkn55_kcal_mol'])} |")
    lines += ['', '单位 kcal/mol；S22 正值表示结合，未做 CP 校正。','',
        '## 优化与规模','', '| 结构 | SZ AO | DZ AO | 总能量降低 SZ→DZ (mEh) | 收敛起点/4 | 终点跨度 (mEh) |',
        '|---|---:|---:|---:|---:|---:|']
    for i,r in dz.items():
        lines.append(f"| {i} | {r.get('sz_nao')} | {r.get('nao')} | {fmt(r.get('free_sz_to_dz_millihartree'))} | {r.get('converged_starts')} | {fmt(r.get('endpoint_spread_millihartree'))} |")
    lines += ['',f'全部起点停止状态：{dict(count)}。',f"运行时间：{report['elapsed_seconds']:.1f} 秒，设备 {report['device']}。",
        '达到步数上限或回退拒绝不等于收敛；SCF/梯度失败的结构不计入完整反应。',
        '三个条目来自前序误差诊断，不能推广为完整 GMTKN55 表现。自由基组是逐结构优化结果，不是 MACE 泛化能力。']
    p=OUT/'gpu4-verification/comparison.json'
    if p.exists():
        v=json.loads(p.read_text());errs=[r['native_energy_error_hartree'] for r in v['results'] if r['status']=='passed']
        lines += ['',f"独立 GPU4PySCF 核验：{v['passed']}/7 通过，{v['failed']} 失败；最大能量差 {max(errs) if errs else None} Eh。"]
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(results=rows,optimizer_statuses=dict(count)),indent=2))

if __name__=='__main__':
    main()
