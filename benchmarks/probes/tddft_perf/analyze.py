"""Summarize saved raw records; standard-library only, no computations rerun."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import pstats
import statistics


def median(values):
    return statistics.median(values) if values else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    records=[];groups={};inputs={};processes={}
    for folder in sorted((root/'runs').iterdir()):
        if not folder.is_dir():continue
        if (folder/'input.json').exists():inputs[folder.name]=json.loads((folder/'input.json').read_text())
        if (folder/'process.json').exists():processes[folder.name]=json.loads((folder/'process.json').read_text())
        path=folder/'measurements.jsonl'
        if not path.exists():continue
        for line in path.read_text().splitlines():
            row=json.loads(line);row['evidence']=str(path.relative_to(root))
            scientific_status=row['status']
            if row['stage']=='molgw_end_to_end' and any(x is None for x in row.get('excitations_ev',[])):
                scientific_status='numerical_mismatch'
            if row['stage']=='failure' and 'No FFI handler registered' in row.get('notes',''):
                scientific_status='unsupported'
            if row['stage']=='failure' and 'SCF did not converge' in row.get('notes',''):
                scientific_status='nonconverged'
            row['scientific_status']=scientific_status
            caveat=('CPU case; GPU memory not applicable.' if row['device']=='cpu' else
                    'GPU lifetime counters are separate; stage-specific VRAM peak unknown.')
            if row.get('gradient_error') is None:caveat+=' No derivative error measured in this row.'
            if row.get('original_operator_residual') is None:caveat+=' Operator residual not exposed here; see dedicated diagnostics.'
            row['measurement_notes']=caveat
            records.append(row)
            key=(folder.name,row['stage'],row.get('batch_size'),row.get('sample'),row.get('rank'),row.get('grid_quality'))
            groups.setdefault(key,[]).append(row)
    with (root/'measurements.jsonl').open('w') as stream:
        for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
    summary=[]
    for (folder,stage,batch,sample,rank,quality),rows in groups.items():
        if stage=='reference_solver_diagnostic':continue
        valid=[x for x in rows if x['wall_seconds'] is not None]
        if not valid:continue
        already_warm=('warm' in stage or stage in {'shared_gradscf_action','shared_pyscf_action'})
        warm=valid if already_warm else [x for x in valid if x.get('repeat',0)>0]
        # Warm-only stages already exclude their first execution. Stages with
        # one record are not relabeled as a warm distribution.
        chosen=warm if warm else valid
        samples=[x['wall_seconds'] for x in chosen]
        middle=median(samples)
        row=rows[-1];cfg=inputs[folder]
        context=next((x for x in records if x['evidence'].split('/')[1]==folder and x['stage']=='reference_metadata'),row)
        runtime_path=root/'runs'/folder/'runtime.json'
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        memory_scope='whole process including compilation and all repeats'
        if cfg['engine']=='pyscf':
            memory_scope=('pure PySCF plus standard harness' if runtime.get('jax_imported') is False
                          else 'harness includes JAX/provenance imports; pure PySCF peak unknown')
        summary.append(dict(run=folder,case_id=row['case_id'],variant=row['variant'],engine=row['engine'],
            stage=stage,batch_size=batch,sample=sample,rank=rank,grid_quality=quality,device=row['device'],threads=cfg['threads'],
            nao=context.get('nao'),nocc=json.dumps(context.get('nocc')),nvir=json.dumps(context.get('nvir')),
            naux=context.get('naux'),ngrid=context.get('ngrid'),nroots=row.get('nroots'),dtype=row.get('dtype'),
            xc=cfg.get('xc'),integral_representation=row.get('integral_representation'),
            sample_count=len(samples),median_seconds=middle,min_seconds=min(samples),max_seconds=max(samples),
            mad_seconds=median([abs(v-middle) for v in samples]),samples_json=json.dumps(samples),
            cpu_peak_rss_bytes=processes.get(folder,{}).get('cpu_peak_rss_bytes'),
            memory_scope=memory_scope,
            status=next((x['scientific_status'] for x in rows if x['scientific_status']!='ok'),processes.get(folder,{}).get('status','partial')),
            evidence=row['evidence']))
    if summary:
        with (root/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    comparisons=[]
    tasks={}
    controls=('atom','basis','xc','spin','charge','representation','auxbasis','grid_level','full','roots','threads','device')
    for folder,cfg in inputs.items():
        if cfg.get('mode')!='end_to_end' or cfg.get('profile') or cfg.get('experiment'):continue
        if 'clean' not in cfg['variant'] and 'cpu8-end_to_end' not in cfg['variant'] and 'cpu8-end_to_end' not in cfg['variant']:continue
        vals=[r for r in records if r['evidence'].split('/')[1]==folder and r['stage']=='end_to_end']
        if not vals:continue
        key=tuple(json.dumps(cfg.get(k,0 if k in ('spin','charge') else None),sort_keys=True) for k in controls)
        tasks.setdefault(key,{})[cfg['engine']]=(folder,vals)
    for engines in tasks.values():
        if set(engines)<{'gradscf','pyscf'}:continue
        g,gr=engines['gradscf'];p,pr=engines['pyscf'];ga,pa=gr[-1],pr[-1]
        error=max(abs(a-b) for a,b in zip(ga['excitations_ha'],pa['excitations_ha']))*27.211386245988
        converged=all(r['status']=='ok' for r in gr+pr)
        metadata={name:[r for r in records if r['evidence'].split('/')[1]==name and r['stage']=='reference_metadata'][-1]
                  for name in (g,p)}
        energy_error=abs(metadata[g]['scf_energy']-metadata[p]['scf_energy'])
        same_dimensions=all(metadata[g].get(k)==metadata[p].get(k) for k in ('nao','nocc','nvir','ngrid'))
        passed=converged and error<=1e-4 and energy_error<=1e-7 and same_dimensions and len(ga['excitations_ha'])==len(pa['excitations_ha'])
        gt=median([r['wall_seconds'] for r in gr if r['repeat']>0]);pt=median([r['wall_seconds'] for r in pr if r['repeat']>0])
        comparisons.append(dict(case_id=ga['case_id'],gradscf_run=g,pyscf_run=p,max_excitation_error_eV=error,
            converged=converged,excitation_gate_passed=passed,scf_energy_error_Ha=energy_error,
            same_dimensions_and_grid_count=same_dimensions,gradscf_warm_seconds=gt,pyscf_warm_seconds=pt,
            time_ratio=gt/pt if passed and gt is not None and pt is not None else None,
            notes='same task CPU end-to-end; algorithm/guard roots differ; grid count and total energy gate must also be inspected'))
    (root/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    compiles=[]
    for path in (root/'runs').glob('*/profile-*.pstats'):
        stats=pstats.Stats(str(path));matches=[]
        for (filename,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
            if name=='backend_compile_and_load':matches.append(dict(file=filename,line=line,calls=nc,own_seconds=tt,cumulative_seconds=ct))
        compiles.append(dict(evidence=str(path.relative_to(root)),backend_compiles=sum(x['calls'] for x in matches),
                            backend_compile_cumulative_seconds=sum(x['cumulative_seconds'] for x in matches),functions=matches))
    (root/'compile_counts.json').write_text(json.dumps(compiles,indent=2)+'\n')
    print(json.dumps(dict(records=len(records),runs=len(inputs),summaries=len(summary),comparisons=comparisons,compiles=compiles),indent=2))


if __name__=='__main__':main()
