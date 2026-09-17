"""Summarize completed/partial ten-system comparisons without hiding failed cases."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


def summarize(root):
    rows=[]
    for path in sorted(root.glob('*/results.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    counts=Counter(r['status'] for r in rows)
    fields=['system','method','xc','status','gradscf_energy_hartree','pyscf_energy_hartree','abs_energy_error_hartree','energy_tolerance_hartree','gradscf_converged','pyscf_converged','gradscf_cycles','pyscf_cycles','gradscf_seconds','pyscf_seconds','density_max_error','gradscf_spin_z','pyscf_spin_z','error']
    with (root/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:
            g,p=r.get('gradscf',{}),r.get('pyscf',{})
            writer.writerow(dict(system=r['system'],method=r['method'],xc=r['xc'],status=r['status'],
                gradscf_energy_hartree=g.get('energy'),pyscf_energy_hartree=p.get('energy'),
                abs_energy_error_hartree=r.get('abs_energy_error'),energy_tolerance_hartree=r.get('energy_tolerance'),
                gradscf_converged=g.get('converged'),pyscf_converged=p.get('converged'),
                gradscf_cycles=g.get('cycles'),pyscf_cycles=p.get('cycles'),
                gradscf_seconds=r.get('gradscf_seconds'),pyscf_seconds=r.get('pyscf_seconds'),
                density_max_error=r.get('density_max_error'),gradscf_spin_z=r.get('gradscf_spin_z'),
                pyscf_spin_z=r.get('pyscf_spin_z'),error=r.get('error',r.get('reason',''))))
    by_system=defaultdict(list);by_method=defaultdict(list)
    for r in rows:
        by_system[r['system']].append(r);by_method[(r['method'],r['xc'])].append(r)
    def stats(group):
        errors=[r['abs_energy_error'] for r in group if r.get('abs_energy_error') is not None]
        converged_errors=[r['abs_energy_error'] for r in group if r.get('abs_energy_error') is not None and r.get('gradscf',{}).get('converged') and r.get('pyscf',{}).get('converged')]
        c=Counter(r['status'] for r in group)
        return dict(counts=dict(c),max_abs_energy_error=max(errors) if errors else None,
                    max_converged_energy_error=max(converged_errors) if converged_errors else None)
    result=dict(counts=dict(counts),systems={k:stats(v) for k,v in by_system.items()},
        methods={m+'/'+x:stats(v) for (m,x),v in by_method.items()},
        failed_cases=[r for r in rows if r['status'] not in {'pass','not_applicable'}])
    result['nonconvergence_sides']=dict(Counter(
        'both' if not r['gradscf']['converged'] and not r['pyscf']['converged'] else
        'gradscf_only' if not r['gradscf']['converged'] else 'pyscf_only'
        for r in rows if r['status']=='not_converged'))
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Ten-system HF/DFT comparison','',f'Recorded rows: {len(rows)}. Status counts: {dict(counts)}.','',
        'Energy errors are absolute Hartree differences. The matrix uses matched geometry, Cartesian 3-21G and common level-1 grid coordinates/weights. Integrals and grid AOs are independently evaluated. Shared hcore density guesses are followed by independent SCF iterations.', '',
        'Pass thresholds: HF 1e-8 Ha, DFT 1e-6 Ha; both SCFs must converge and conserve electron number to 1e-6. Density differences are diagnostic because degenerate orbitals/spin states can differ. No finite-difference or parameter optimization is performed.', '',
        '| System | Passed | Failed/error/nonconverged | Not applicable | Max energy error / Ha |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,group in sorted(by_system.items()):
        s=stats(group);c=s['counts'];error=s['max_abs_energy_error']
        text.append(f"| {name} | {c.get('pass',0)} | {sum(v for k,v in c.items() if k not in {'pass','not_applicable'})} | {c.get('not_applicable',0)} | {error:.3e} |" if error is not None else f'| {name} | 0 | {len(group)} | 0 | n/a |')
    text += ['', '| Method / XC | Passed | Failed/error/nonconverged | Max energy error / Ha |','| --- | ---: | ---: | ---: |']
    for key,group in sorted(by_method.items()):
        s=stats(group);c=s['counts'];error=s['max_abs_energy_error'];error_text=f'{error:.3e}' if error is not None else 'n/a'
        text.append(f"| {'/'.join(key)} | {c.get('pass',0)} | {sum(v for k,v in c.items() if k not in {'pass','not_applicable'})} | {error_text} |")
    text += ['', '## Cases requiring investigation','']
    for r in result['failed_cases']:
        text.append(f"- {r['system']} {r['method']}/{r['xc']}: {r['status']}; energy error={r.get('abs_energy_error')}; {r.get('error','')}")
    if not result['failed_cases']:text.append('None among recorded rows. Check the root summary for expected row count and worker completion before claiming full coverage.')
    text += ['', 'Results are for fixed test geometries and this basis/grid. Unrestricted/generalized solutions can occupy different SCF basins. Restricted closed-shell methods are not applied to open-shell systems. MGGA, range-separated SCF, multicollinear GKS and response/excited-state calculations are outside the implemented classic matrix. Timings include per-worker compilation and are not a controlled speed benchmark.']
    (root/'README.md').write_text('\n'.join(text)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    result=summarize(args.root)
    print(json.dumps(result['counts'],indent=2))
