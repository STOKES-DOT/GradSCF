"""Freeze the reviewed experiment definition and generate its Slurm inputs."""
import json
from pathlib import Path
import subprocess

from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash

ROOT=Path(__file__).resolve().parent
PROJECT=repository_root()
REMOTE='/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021'
LANES=dict(szp=('s','m'),dzp=('s','m','l'),tzp=('s','m','l'))
inputs=[f'inputs/{n}.json' for n in ('references','pilot','model-config','s-candidate','l-candidate')]
data=json.loads((ROOT/'inputs/references.json').read_text())
assert len(data['structures'])==500 and len({r['structure_id'] for r in data['structures']})==500
assert all(len(r['symbols'])<=30 and r['charge']==r['spin']==0 for r in data['structures'])
atomic_json(ROOT/'campaign.json',dict(scientific_source_sha256=scientific_source_hash(PROJECT),
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
    input_sha256={name:file_hash(ROOT/name) for name in inputs},
    lanes={z:[f'nnao_{z}_{t}' for t in ts] for z,ts in LANES.items()},
    existing_family='nnao_szp_l',max_simultaneous_gpus=3,
    seed=0,batch_size=50,micro_batch=1,updates_per_epoch=10,epochs=1000,
    learning_rate=1e-3,decay_epochs=100,decay_factor=.5,
    early_stopping=dict(window=10,tolerance_hartree=1e-4,metric='sum_absolute_online_mean_changes'),
    pool_provenance=dict(s='GradSCF_pool_calibration_gpu_20261014 / main-group1000-core-sp6-d3-20261013 / nnao_dzp_s',
        m='26-reference DZ pool calibration; 20 exponent steps, 3 starts x300 contraction steps',
        l='GradSCF_mace500_20261014 / mace500-20261014 / selection/candidate.json')))


def script(name,gpu,hours,command,*,node=None,memory=512):
    lines=['#!/bin/bash','#SBATCH --partition=home',f'#SBATCH --job-name={name}',
        f'#SBATCH --gres=gpu:{gpu}:1','#SBATCH --cpus-per-task=32',f'#SBATCH --mem={memory}G',
        f'#SBATCH --time={hours}:00:00',f'#SBATCH --chdir={REMOTE}',
        f'#SBATCH --output=artifacts/nnao-family500-20261021/{name}-%j.log']
    if node:lines.append(f'#SBATCH --nodelist={node}')
    lines.extend(['set -euo pipefail',
        'export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH',
        'export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax',
        'export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4',
        f'export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/{name}',
        'export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0','date -Is',command,'date -Is'])
    (ROOT/(name+'.slurm')).write_text('\n'.join(lines)+'\n')


script('prepare','A800',4,'''export CC=/usr/bin/gcc CXX=/usr/bin/g++
"$CXX" --version
JAX_PLATFORMS=cpu python -m gradscf.integrals._native.build --jobs 8 --build-dir artifacts/nnao-family500-20261021/native-build
JAX_PLATFORMS=cpu python -m pytest -q tests/nnao/test_energy_stopping.py tests/nnao/test_minibatch_training.py tests/nnao/test_single_pass_training.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_pool_selection_checks.py -k 'not real_mace'
python -u artifacts/nnao-family500-20261021/campaign.py prepare''',node='n005')
for zeta,tiers in LANES.items():
    for tier in tiers:
        family=f'nnao_{zeta}_{tier}'
        script(family,'A800' if zeta=='szp' else 'A100_80G',72,
            f'python -u artifacts/nnao-family500-20261021/campaign.py train --family {family}',
            node='n004' if zeta=='szp' else None,memory=448 if zeta=='szp' else 512)
definition=json.loads((ROOT/'campaign.json').read_text())
definition['workflow_sha256']={p.name:file_hash(p) for p in
    [ROOT/'campaign.py',ROOT/'submit.py',*sorted(ROOT.glob('*.slurm'))]}
atomic_json(ROOT/'campaign.json',definition)
print(json.dumps(dict(status='prepared',families=sum(map(len,LANES.values())),
    source_sha256=json.loads((ROOT/'campaign.json').read_text())['scientific_source_sha256'])))
