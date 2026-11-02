"""Nine fresh 1000-system runs in three serial GPU lanes."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
jobs = []
for lane in ('szp','dzp','tzp'):
    for tier in ('s','m','l'):
        family = f'nnao_{lane}_{tier}'
        gpu = 'A800' if lane=='szp' else 'A100_80G'
        node = '#SBATCH --nodelist=n004\n' if lane=='szp' else ''
        text = f'''#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao1000-{lane}-{tier}
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
{node}#SBATCH --output=artifacts/main-group1000-restart-20261013/{family}-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/main-group1000-restart-20261013/jit-cache/{family}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -c 'import json; from pathlib import Path; from gradscf.model.nnao.runtime import scientific_source_hash, repository_root; p=Path("artifacts/main-group1000-restart-20261013"); d=json.loads((p/"validation-passed.json").read_text()); h=scientific_source_hash(repository_root()); assert d["passed"] and len(d["results"])==9 and all(r["scientific_source_sha256"]==h for r in d["results"]), "Validation/source mismatch"'
python -u -m gradscf.model.nnao.workflows.basis_grid run --family {family} --references artifacts/main-group1000-restart-20261013/inputs/references.json --pilot-references artifacts/main-group1000-restart-20261013/inputs/pilot.json --output-dir artifacts/main-group1000-restart-20261013/runs/{family} --epochs 1000 --calibration-steps 20 --contraction-steps 300 --starts 3
'''
        name = f'{family}.slurm'
        (root/name).write_text(text)
        jobs.append(dict(family=family,lane=lane,gpu=gpu,cpus=32,memory_gib=256,hours=72,
                         structures=1000,epochs=1000,script=f'artifacts/main-group1000-restart-20261013/{name}'))
(root/'job-plan.json').write_text(json.dumps(dict(validation_job=96691,retained_original_job=96510,
    max_simultaneous_gpus=3,max_simultaneous_cpus=96,fresh_MACE=True,reset_Adam=True,jobs=jobs),indent=2)+'\n')
print('Prepared 9 jobs, each with 1000 structures and 1000 target epochs.')
