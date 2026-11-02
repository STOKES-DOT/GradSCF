"""Eight fixed-pool runs using the approved 500-system training configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash

ROOT=Path(__file__).resolve().parent
LANES=dict(szp=('s','m'),dzp=('s','m','l'),tzp=('s','m','l'))
FAMILIES=tuple(f'nnao_{z}_{t}' for z,tiers in LANES.items() for t in tiers)


def read(name):
    return json.loads((ROOT/name).read_text())


def verify():
    manifest=read('campaign.json')
    if scientific_source_hash(repository_root())!=manifest['scientific_source_sha256']:
        raise ValueError('Scientific source differs from this campaign.')
    for name,expected in (manifest['input_sha256']|manifest['workflow_sha256']).items():
        if file_hash(ROOT/name)!=expected:
            raise ValueError('Campaign input changed: '+name)
    return manifest


def configuration(family):
    data=dict(read('inputs/references.json'),basis_family=family)
    model=dict(read('inputs/model-config.json'),basis_family=family)
    return data,model


def train(family,*,smoke=False):
    import jax
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    verify()
    if jax.default_backend()!='gpu' or len(jax.devices('gpu'))!=1:
        raise RuntimeError('This campaign requires exactly one Slurm-assigned GPU per job.')
    data,model=configuration(family)
    tier=family.rsplit('_',1)[1]
    candidate=read(f'inputs/{tier}-candidate.json')
    if smoke:
        data=dict(data,structures=[next(r for r in read('inputs/pilot.json')['structures']
                                      if r['structure_id']=='H2')])
    else:
        ready=read('ready.json')
        if ready['status']!='complete' or ready['candidate_sha256'][tier]!=file_hash(ROOT/f'inputs/{tier}-candidate.json'):
            raise ValueError('Pool preparation or numerical smoke gate is incomplete.')
    output=ROOT/'runs'/family/('smoke' if smoke else 'train')
    progress=output/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['status'] in {'complete','early_stopped'}:
        print('Already completed:',family,flush=True)
        return
    options={} if smoke else dict(early_stop_window=10,early_stop_tolerance=1e-4)
    result=run_fixed_pool(data,candidate,ROOT/'cache'/tier,output,
        epochs=1 if smoke else 1000,learning_rate=1e-3,seed=0,storage='mmap',
        resume=(output/'last.npz').exists(),model_config=model,
        batch_size=1 if smoke else 50,decay_epochs=100,check_parameter_gradients=False,**options)
    if result['status'] not in {'complete','early_stopped'}:
        raise RuntimeError('Training did not finish: '+result['status'])
    atomic_json(output/'result.json',dict(status=result['status'],epoch=result['epoch'],
        update_step=result['update_step'],loss_hartree=result['loss_hartree'],
        loss_kind=result['loss_kind'],parameters_evaluated=result['parameters_evaluated']))


def calibrate():
    import jax
    from gradscf.model.nnao.workflows.pool_calibration import run_pilot
    from gradscf.model.nnao.workflows.basis_grid import select_candidate
    verify()
    if jax.default_backend()!='gpu' or len(jax.devices('gpu'))!=1:
        raise RuntimeError('Pool calibration requires exactly one Slurm-assigned GPU.')
    pilot=dict(read('inputs/pilot.json'),basis_family='nnao_dzp_m')
    target=ROOT/'inputs/m-candidate.json'
    if not target.exists():
        directory=ROOT/'m-calibration'
        summary=directory/'summary.json'
        if not summary.exists() or json.loads(summary.read_text())['status']!='complete':
            run_pilot(pilot,directory,calibration_steps=20,contraction_steps=300,
                      starts=3,seed=0,check_gradients=False,scf_platform='cuda')
        atomic_json(target,select_candidate(directory,pilot,require_gradient_check=False))


def prepare():
    import jax
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.initialization import bind_pool_candidate
    verify()
    subprocess.run([sys.executable,'-u',str(Path(__file__)),'calibrate'],check=True)
    pilot=read('inputs/pilot.json')
    for family in FAMILIES:
        candidate=read(f'inputs/{family.rsplit("_",1)[1]}-candidate.json')
        with jax.default_device(jax.devices('cpu')[0]):
            for row in pilot['structures']:
                layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                    basis_family=family,core_primitives='6-31g')
                bind_pool_candidate(layout,candidate)
        # Each process releases its full-size MACE and SCF compilation state.
        subprocess.run([sys.executable,'-u',str(Path(__file__)),'smoke','--family',family],check=True)
    atomic_json(ROOT/'ready.json',dict(status='complete',
        scientific_source_sha256=scientific_source_hash(repository_root()),
        candidate_sha256={t:file_hash(ROOT/f'inputs/{t}-candidate.json') for t in ('s','m','l')},
        smoke_families=list(FAMILIES),finite_differences=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','calibrate','train','smoke'))
    parser.add_argument('--family',choices=FAMILIES)
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare()
    elif args.mode=='calibrate':
        calibrate()
    elif args.family is None:
        parser.error('--family is required for train/smoke')
    else:
        train(args.family,smoke=args.mode=='smoke')
