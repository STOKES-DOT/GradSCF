#!/usr/bin/env python3
"""Thin CPU worker: fixed-reference tables, train-only fit and audited records.

External design modules define get_candidate(id) -> dict with terms,
pointwise(theta,rho_spin,grad_spin), initial, bounds, formula and references.
Luna proposes formulas; the parent reviews/implements the module and runs this.
All integrals, potentials and fitting objectives are shared src APIs.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from gradscf_tools.iks_workflow import file_hash, select_references
from gradscf_tools.pbe_loop import (build_single_pbe_input,build_molecular_affine_cache,
    save_affine_cache,load_affine_cache,load_teacher_reference,summarize_affine)
from gradscf_tools.pbe_experiment import (CONSTRAINT_NAMES,fit_and_score)
from gradscf_tools.pbe_recording import (canonical_hash,freeze_reward_config,write_candidate_bundle)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as handle:
        handle.write(json.dumps(data,indent=2,allow_nan=False)+'\n')


def design(path, candidate_id):
    if candidate_id == 'c000':
        import jax.numpy as jnp
        return dict(terms=(),initial=[],bounds=None,
            pointwise=lambda theta,rho,grad:jnp.zeros_like(rho[0]),
            formula='# c000: unmodified PBE\n\nE_xc = E_x^PBE + E_c^PBE. No fitted parameters.\n',
            references=['https://doi.org/10.1103/PhysRevLett.77.3865'])
    if path is None:
        raise ValueError('An external reviewed design module is required')
    spec=importlib.util.spec_from_file_location('external_pbe_design',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_candidate(candidate_id)


def cache_one(args):
    teacher=load_teacher_reference(args.teacher)
    candidate=design(args.module,args.candidate)
    baseline=None if args.baseline is None else load_affine_cache(args.baseline)
    state,enuc=build_single_pbe_input(teacher,max_memory_mb=6000.,with_coulomb=baseline is None)
    cache=build_molecular_affine_cache(teacher,state,candidate['terms'],nuclear_repulsion=enuc,
                                      baseline_cache=baseline)
    saved=save_affine_cache(cache,args.output,teacher_json=args.teacher,metadata=dict(
        candidate_id=args.candidate,module_sha256=None if args.module is None else file_hash(args.module)))
    print(json.dumps(saved),flush=True)


def tables(args):
    references=json.loads(Path(args.references).read_text())
    rows=select_references(references)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    baseline={} if args.baseline is None else {r['molecule_id']:r for r in
        json.loads(Path(args.baseline).read_text())['rows']}
    results=[]
    for row in rows:
        name=row['molecule_id']
        if file_hash(row['teacher_json']) != row['teacher_json_sha256']:
            raise ValueError('Reference JSON changed after qualification')
        target=output/(name+'.npz')
        command=[sys.executable,__file__,'cache','--candidate',args.candidate,
            '--teacher',row['teacher_json'],'--output',str(target)]
        if args.module is not None:
            command += ['--module',args.module]
        if args.baseline is not None:
            previous=baseline[name]
            if file_hash(previous['cache']) != previous['sha256']:
                raise ValueError('Baseline cache changed')
            command += ['--baseline',previous['cache']]
        started=time.perf_counter()
        with (output/(name+'.out')).open('w') as stdout, (output/(name+'.err')).open('w') as stderr:
            completed=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
        if completed.returncode or not target.is_file():
            raise RuntimeError(f'Fixed-reference table preparation failed for {name}; see logs')
        results.append(dict(molecule_id=name,split=row['split'],teacher_json=row['teacher_json'],
            teacher_json_sha256=row['teacher_json_sha256'],cache=str(target),sha256=file_hash(target),
            elapsed_seconds=time.perf_counter()-started))
        print(json.dumps(dict(molecule_id=name,completed=len(results),total=len(rows))),flush=True)
    write_json(output/'tables.json',dict(status='qualified',candidate_id=args.candidate,
        reference_manifest=args.references,reference_manifest_sha256=file_hash(args.references),
        protocol_sha256=references['protocol_sha256'],
        module_sha256=None if args.module is None else file_hash(args.module),rows=results))


def assess(args):
    manifest=json.loads(Path(args.tables).read_text())
    if manifest['candidate_id'] != args.candidate or manifest['status'] != 'qualified':
        raise ValueError('Candidate and qualified table manifest must agree')
    if manifest['module_sha256'] != (None if args.module is None else file_hash(args.module)):
        raise ValueError('External form changed after table construction')
    if file_hash(manifest['reference_manifest']) != manifest['reference_manifest_sha256']:
        raise ValueError('Reference manifest changed')
    caches={'train':[],'validation':[]}
    for row in manifest['rows']:
        if row['split'] not in caches:
            raise ValueError('Held-out test cannot enter the adaptive loop')
        caches[row['split']].append(load_affine_cache(row['cache'],expected_sha256=row['sha256']))
    if {role:len(v) for role,v in caches.items()} != {'train':20,'validation':5}:
        raise ValueError('Require the predeclared20/5 adaptive split')
    candidate=design(args.module,args.candidate)
    output=Path(args.output)
    if args.candidate == 'c000':
        baseline=summarize_affine(caches['validation'],[])
        reward=freeze_reward_config(baseline['energy_wrmsd_hartree'],constraint_names=CONSTRAINT_NAMES,
            provenance=dict(reference_protocol_sha256=manifest['protocol_sha256'],
                table_manifest_sha256=file_hash(args.tables),objective='MSE+MAE+.01*rotation_norm2',
                energy_scale_hartree=1.,gap_floor_hartree=.05))
        write_json(output/'reward.json',reward)
    else:
        reward=json.loads((output/'reward.json').read_text())
        if reward['provenance']['reference_protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Reward and reference protocol must agree')
    scored=fit_and_score(caches,initial=candidate['initial'],bounds=candidate.get('bounds'),
        reward_config=reward,pointwise=candidate['pointwise'],
        fixed_parameters=[] if args.candidate=='c000' else None)
    predictions=scored.pop('predictions')
    feedback=(f"# Measured feedback for {args.candidate}\n\n"
        'Fixed iKS references; no candidate SCF. Units: Hartree. Test remains sealed.\n\n'
        +json.dumps(scored,indent=2)+'\n')
    record=dict(candidate_id=args.candidate,status='evaluated',parameters=scored['parameters'],
        initial=candidate['initial'],bounds=candidate.get('bounds'),scores=scored,
        references=candidate['references'],module_sha256=manifest['module_sha256'],
        table_manifest_sha256=file_hash(args.tables),reference_protocol_sha256=manifest['protocol_sha256'],
        reward_sha256=reward['sha256'],training_method='fixed iKS density/orbitals; affine scalar-energy AD',
        inverse_approximation='relative density L2 <=1e-3; not an exact-potential benchmark')
    folder=write_candidate_bundle(output/'candidates',record,formula=candidate['formula'],
        feedback=feedback,predictions=predictions)
    print(json.dumps(dict(folder=str(folder),metrics=scored['metrics'],reward=scored['reward'])),flush=True)


def heldout(args):
    """One final baseline/winner evaluation; no fitting or adaptive reward."""
    winner=json.loads(Path(args.winner).read_text())
    body={k:v for k,v in winner.items() if k!='sha256'}
    if winner.get('sha256')!=canonical_hash(body):
        raise ValueError('Frozen winner hash changed')
    references=json.loads(Path(args.references).read_text())
    rows=select_references(references,splits=('test',),frozen_winner=winner)
    if len(rows)!=5:
        raise ValueError('Require the five sealed test references')
    module_hash=None if args.module is None else file_hash(args.module)
    if module_hash!=winner['module_sha256']:
        raise ValueError('Winner formula changed after validation selection')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'test_opening.json',dict(frozen_winner_sha256=winner['sha256'],
        protocol_sha256=winner['protocol_sha256'],policy='No fitting or subsequent model selection'))
    caches={'baseline':[],'winner':[]}
    predictions=[]
    for row in rows:
        if file_hash(row['teacher_json'])!=row['teacher_json_sha256']:
            raise ValueError('Test teacher changed after qualification')
        base_path=output/(row['molecule_id']+'-baseline.npz')
        for role,candidate,parameters in (('baseline','c000',[]),
                ('winner',winner['candidate_id'],winner['parameters'])):
            path=base_path if role=='baseline' else output/(row['molecule_id']+'-winner.npz')
            if role=='winner' and candidate=='c000':
                cache=caches['baseline'][-1]
            else:
                command=[sys.executable,__file__,'cache','--candidate',candidate,
                    '--teacher',row['teacher_json'],'--output',str(path)]
                if role=='winner':
                    command+=['--module',args.module,'--baseline',str(base_path)]
                with path.with_suffix('.out').open('w') as stdout, path.with_suffix('.err').open('w') as stderr:
                    completed=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
                if completed.returncode:
                    raise RuntimeError('Held-out evaluation failed; preserve the frozen winner and logs')
                cache=load_affine_cache(path)
            caches[role].append(cache)
            value=cache.evaluate(parameters)
            predictions.append(dict(molecule_id=row['molecule_id'],model_role=role,
                candidate_id=candidate,reference=float(cache.teacher.target.energy),
                prediction=float(value.total_energy),signed_error=float(value.energy_error),units='Hartree'))
        print(json.dumps(dict(test_completed=len(caches['baseline']),total=5)),flush=True)
    metrics=dict(baseline=summarize_affine(caches['baseline'],[]),
        winner=summarize_affine(caches['winner'],winner['parameters']))
    record=dict(status='evaluated',frozen_winner=winner,metrics=metrics,predictions=predictions,
        reference_manifest_sha256=file_hash(args.references),scope='Final fixed-reference test; no candidate SCF')
    write_json(output/'test_report.json',record)
    with (output/'predictions.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]))
        writer.writeheader();writer.writerows(predictions)
    print(json.dumps(dict(status='evaluated',metrics=metrics)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('cache','tables','assess'):
        sub=commands.add_parser(name)
        sub.add_argument('--candidate',required=True)
        sub.add_argument('--module')
        sub.add_argument('--output',required=True)
    cache_parser=commands.choices['cache']
    cache_parser.add_argument('--teacher',required=True)
    cache_parser.add_argument('--baseline')
    commands.choices['tables'].add_argument('--references',required=True)
    commands.choices['tables'].add_argument('--baseline',help='Baseline tables.json')
    commands.choices['assess'].add_argument('--tables',required=True)
    held=commands.add_parser('heldout')
    held.add_argument('--winner',required=True)
    held.add_argument('--references',required=True)
    held.add_argument('--module')
    held.add_argument('--output',required=True)
    args=parser.parse_args()
    dict(cache=cache_one,tables=tables,assess=assess,heldout=heldout)[args.command](args)


if __name__=='__main__':
    main()
