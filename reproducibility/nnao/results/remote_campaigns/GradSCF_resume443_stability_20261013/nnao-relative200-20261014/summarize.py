"""Compute signed stoichiometric differences; incomplete reactions stay missing."""
import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
# CODATA Hartree / (kcal mol^-1), consistent across all compared bases.
HARTREE_TO_KCAL_MOL = 627.5094740631
BASES = ['minao', '3-21g', '6-31g(d,p)', 'def2-svp', 'cc-pvdz', 'cc-pvtz',
         'NNAO-200-initial', 'NNAO-200-best']

def relative_energy(reaction, energies):
    return (math.fsum(t['coefficient'] * energies[t['structure_id']]
                      for t in reaction['terms']) * HARTREE_TO_KCAL_MOL
            + reaction['energy_add_kcal_mol'])

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    source = json.loads((OUT/'gpu4/comparison.json').read_text())
    reactions = [json.loads(s) for s in (OUT/'reactions.jsonl').read_text().splitlines()]
    energies = {basis: {r['structure_id']: r['energy_hartree'] for r in source['results']
                       if r['basis'] == basis and r['status'] == 'passed'} for basis in BASES}
    rows = []
    for reaction in reactions:
        ids = {t['structure_id'] for t in reaction['terms']}
        anchor = (relative_energy(reaction, energies['cc-pvtz'])
                  if ids <= energies['cc-pvtz'].keys() else None)
        equation = ' '.join(f"{t['coefficient']:+g}*{t['structure_id']}" for t in reaction['terms'])
        for basis in BASES:
            missing = sorted(ids-energies[basis].keys())
            value = None if missing else relative_energy(reaction, energies[basis])
            rows.append(dict(reaction_id=reaction['reaction_id'], category=reaction['category'],
                basis=basis, status='missing' if missing else 'passed', equation=equation,
                energy_kcal_mol=value, gmtkn55_reference_kcal_mol=reaction['reference_energy_kcal_mol'],
                hf_ccpvtz_kcal_mol=anchor,
                error_vs_gmtkn55_kcal_mol=None if value is None else value-reaction['reference_energy_kcal_mol'],
                error_vs_hf_ccpvtz_kcal_mol=None if value is None or anchor is None else value-anchor,
                missing_structures=';'.join(missing)))
    summary = []
    for category in ['reaction', 'barrier', 'noncovalent']:
        group = [r for r in rows if r['category'] == category]
        complete = {r['reaction_id'] for r in group}
        complete -= {r['reaction_id'] for r in group if r['status'] != 'passed'}
        for basis in BASES:
            valid = [r for r in group if r['basis'] == basis and r['reaction_id'] in complete]
            record = dict(category=category, basis=basis, common_complete_count=len(valid), target_count=6)
            for reference in ['hf_ccpvtz', 'gmtkn55']:
                errors = [r[f'error_vs_{reference}_kcal_mol'] for r in valid]
                record.update({f'{metric}_vs_{reference}_kcal_mol': value for metric, value in [
                    ('mae', sum(map(abs, errors))/len(errors) if errors else None),
                    ('signed_mean', sum(errors)/len(errors) if errors else None),
                    ('max_abs', max(map(abs, errors)) if errors else None)]})
            summary.append(record)
    write_csv(OUT/'relative-energies.csv', rows)
    write_csv(OUT/'category-metrics.csv', summary)
    failures = [{k: r.get(k) for k in ['structure_id', 'basis', 'error_type', 'error']}
                for r in source['results'] if r['status'] != 'passed']
    result = dict(status='complete' if not failures else 'complete_with_failures',
        checkpoint=json.loads((OUT/'snapshot.json').read_text()),
        selection=json.loads((OUT/'selection-manifest.json').read_text()),
        backend=source['backend'], hardware=source['provenance']['gpu_name'],
        cpu_threads=source['cpu_threads'], elapsed_seconds=source['elapsed_seconds'],
        settings=source['settings'], failures=failures, metrics=summary,
        native_gpu4_max_error_hartree=max((r.get('native_energy_error_hartree', 0)
            for r in source['results'] if r['status']=='passed'), default=None),
        limitations=['HF/cc-pVTZ is a finite-basis comparison, not CBS.',
            'GMTKN55 error contains both basis and HF method error.',
            'NCI values are uncorrected for BSSE; independently predicted monomer and complex contractions.',
            'Selected 18-reaction exploratory panel, not full GMTKN55 or WTMAD2.',
            'Category comparisons use the same complete-reaction intersection for every basis.'])
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# NNAO 相对能量测试', '',
        f"固定模型：200 体系训练 epoch {result['checkpoint']['checkpoint_epoch']}；18 个相对能量，39 个结构。",
        '反应能和势垒按原始符号；非共价条目为正值表示结合的解离能，未做 CP 校正。', '',
        '| 类别 | 基组 | 共同有效条目 | MAE vs HF/cc-pVTZ (kcal/mol) | MAE vs GMTKN55 (kcal/mol) |',
        '|---|---|---:|---:|---:|']
    fmt = lambda x: '缺失' if x is None else f'{x:.4f}'
    for r in summary:
        lines.append(f"| {r['category']} | {r['basis']} | {r['common_complete_count']}/6 | {fmt(r['mae_vs_hf_ccpvtz_kcal_mol'])} | {fmt(r['mae_vs_gmtkn55_kcal_mol'])} |")
    lines += ['', f'GPU：{result["hardware"]}；CPU：{result["cpu_threads"]}。',
        f'独立 GPU4PySCF 同基组核对最大误差：{result["native_gpu4_max_error_hartree"]} Eh。',
        '', '## 解释范围', '', *['- '+s for s in result['limitations']], '',
        '详细逐条数值见 relative-energies.csv；失败条目及版本信息见 summary.json。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], failures=failures, metrics=summary), indent=2))

if __name__ == '__main__':
    main()
