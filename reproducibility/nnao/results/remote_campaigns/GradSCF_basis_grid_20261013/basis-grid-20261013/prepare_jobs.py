"""Render eight serial-lane jobs; existing L-SZP is job 96510."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
original = '/home/user_yuanjiao/private/homefile/GradSCF'
jobs = []
for zeta in ('szp', 'dzp', 'tzp'):
    for tier in ('s', 'm', 'l'):
        if (zeta, tier) == ('szp', 'l'):
            continue
        family = f'nnao_{zeta}_{tier}'
        gpu = 'A800' if zeta == 'szp' else 'A100_80G'
        node = '#SBATCH --nodelist=n004\n' if zeta == 'szp' else ''
        candidate = (f' --fixed-candidate {original}/artifacts/fixed-pool200-20261012/candidate-exponents.json'
                     if tier == 'l' else '')
        script = f'''#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name={family}
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
{node}#SBATCH --output=artifacts/basis-grid-20261013/{family}-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/basis-grid-20261013/jit-cache/{family}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -c 'import json; assert json.load(open("artifacts/basis-grid-20261013/validation-passed.json"))["passed"]'
python -c 'import json; from pathlib import Path; from gradscf.model.nnao.runtime import scientific_source_hash, repository_root; h=scientific_source_hash(repository_root()); files=list(Path("artifacts/basis-grid-20261013/smoke").glob("*/training/metadata.json")); assert len(files)==9 and all(json.loads(p.read_text())["identity"]["scientific_source_sha256"]==h for p in files), "Validation source differs from training source"'
python -u -m gradscf.model.nnao.workflows.basis_grid run --family {family} --references {original}/artifacts/fixed-pool200-20261012/references.json --output-dir artifacts/basis-grid-20261013/runs/{family} --epochs 1000 --calibration-steps 20 --contraction-steps 300 --starts 3{candidate}
'''
        (root/f'{family}.slurm').write_text(script)
        jobs.append(dict(family=family, lane=zeta, gpu=gpu, cpus=32, memory_gib=256,
                         hours=72, script=f'artifacts/basis-grid-20261013/{family}.slurm'))
(root/'job-plan.json').write_text(json.dumps(dict(existing_szp_l_job=96510,
    validation_job=96631, max_simultaneous_gpus=3, max_simultaneous_cpus=96,
    jobs=jobs), indent=2)+'\n')
print(f'Rendered {len(jobs)} jobs; submission is a separate operation.')
