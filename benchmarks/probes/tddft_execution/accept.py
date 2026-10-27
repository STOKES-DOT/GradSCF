"""Offline numerical gates and paired timing summaries, after clean runs.

No original measurement/process file is rewritten. Kernel gates compare the
saved nonsymmetric-density outputs; molecular gates additionally require the
controller's original-operator residual certificate. GPU CPU8 vs CPU1 times
are deliberately not paired. Energy thresholds are Ha/eV as stated below.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

ENERGY_EV = 1e-4
SCF_HA = 1e-7
OSCILLATOR_ABS = 1e-7
PACKED_ABS = 3e-11


def rows(folder):
    return [json.loads(s) for s in (folder/'measurements.jsonl').read_text().splitlines()]


def median(folder, stage):
    return statistics.median(x['wall_seconds'] for x in rows(folder)
                             if x['stage']==stage and x.get('repeat', 0)>0)


def csv_file(path, data):
    if data:
        with path.open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('evidence',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paired=[];kernels=[];actions=[]
    for parent_group, final_group in [('parent','final'),('parent-cpu8','final-cpu8'),
                                      ('parent','final-verified')]:
        for final in sorted((args.evidence/final_group).glob('*/process.json')):
            final=final.parent;cfg=json.loads((final/'input.json').read_text())
            candidates=list((args.evidence/parent_group).glob(cfg['case_id']+'-*/process.json'))
            if len(candidates)!=1 or cfg['device']!='cpu':continue
            parent=candidates[0].parent
            pr,fr=rows(parent),rows(final)
            pe=next(x for x in reversed(pr) if x['stage']=='end_to_end')
            fe=next(x for x in reversed(fr) if x['stage']=='end_to_end')
            error=float(np.max(np.abs(np.asarray(pe['excitations_ha'])-fe['excitations_ha']))*27.211386245988)
            scf_error=abs(next(x['scf_energy'] for x in pr if x['stage']=='reference_metadata')
                          -next(x['scf_energy'] for x in fr if x['stage']=='reference_metadata'))
            cuts=[0]+(np.flatnonzero(np.diff(pe['excitations_ha'])>1e-5)+1).tolist()+[len(pe['excitations_ha'])]
            strength_error=max(abs(sum(pe['oscillator_strength'][a:b])-sum(fe['oscillator_strength'][a:b]))
                               for a,b in zip(cuts[:-1],cuts[1:]))
            process=[json.loads((p/'process.json').read_text()) for p in (parent,final)]
            passed=all(x['accepted_for_comparison'] for x in process) and error<=ENERGY_EV and scf_error<=SCF_HA and strength_error<=OSCILLATOR_ABS
            for stage in ('response_solve','end_to_end'):
                before,after=median(parent,stage),median(final,stage)
                paired.append(dict(case_id=cfg['case_id'],threads=cfg['threads'],stage=stage,
                    parent_run=str(parent.relative_to(args.evidence)),final_run=str(final.relative_to(args.evidence)),
                    before_seconds=before,after_seconds=after,
                    time_ratio=after/before if passed else None,accepted_for_comparison=passed,
                    max_excitation_error_ev=error,scf_error_ha=scf_error,cluster_strength_error=strength_error))
    for path in sorted((args.evidence/'kernel-final').glob('*/values.npz')):
        folder=path.parent;cfg=json.loads((folder/'input.json').read_text())
        parent=args.evidence/'kernel-parent'/f"packed-{cfg['nao']}-{cfg['device']}-td-private-parent"
        if not (parent/'values.npz').exists():continue
        a,b=np.load(parent/'values.npz'),np.load(path)
        error=max(float(np.max(np.abs(a[key]-b[key]))) for key in ('j','k'))
        before,after=median(parent,'warm_action'),median(folder,'warm_action')
        processes=[json.loads((p/'process.json').read_text()) for p in (parent,folder)]
        passed=error<=PACKED_ABS and all(x['process_status']=='ok' for x in processes)
        memories=[next(x['compiler_memory']['temp_size_in_bytes'] for x in rows(p) if x['stage']=='compile')
                  for p in (parent,folder)]
        kernels.append(dict(case_id=cfg['case_id'],parent_run=str(parent.relative_to(args.evidence)),
            final_run=str(folder.relative_to(args.evidence)),device=cfg['device'],nao=cfg['nao'],
            batch=cfg['batch'],max_memory_mb=cfg.get('max_memory',64.),block_size=cfg.get('block_size'),
            before_seconds=before,after_seconds=after,time_ratio=after/before if passed else None,
            max_value_error=error,accepted_for_comparison=passed,
            before_xla_temp_bytes=memories[0],after_xla_temp_bytes=memories[1]))
    csv_file(args.output/'before_after.csv',paired)
    csv_file(args.output/'packed_comparisons.csv',kernels)
    parents=list((args.evidence/'action-parent').glob('*/action-batch6.npz'))
    finals=list((args.evidence/'action-final').glob('*/action-batch6.npz'))
    if len(parents)==len(finals)==1:
        parent,final=parents[0].parent,finals[0].parent
        for batch in (6,12):
            a,b=[np.load(p/f'action-batch{batch}.npz') for p in (parent,final)]
            error=float(np.max(np.abs(a['action']-b['action'])))
            samples=[[x['wall_seconds'] for x in rows(p) if x['stage']=='action_warm'
                      and x['batch_size']==batch] for p in (parent,final)]
            passed=bool(np.array_equal(a['x'],b['x']) and error<=PACKED_ABS)
            before,after=map(statistics.median,samples)
            actions.append(dict(batch=batch,before_seconds=before,after_seconds=after,
                time_ratio=after/before if passed else None,max_value_error=error,
                accepted_for_comparison=passed,samples_seconds=samples,
                parent_run=str(parent.relative_to(args.evidence)),final_run=str(final.relative_to(args.evidence))))
    (args.output/'action_comparisons.json').write_text(json.dumps(actions,indent=2)+'\n')
    (args.output/'acceptance.json').write_text(json.dumps(dict(energy_tolerance_ev=ENERGY_EV,
        scf_tolerance_ha=SCF_HA,oscillator_cluster_absolute_tolerance=OSCILLATOR_ABS,
        packed_absolute_tolerance=PACKED_ABS,molecular=paired,kernels=kernels,actions=actions),indent=2)+'\n')


if __name__=='__main__':main()
