#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=gradscf-xc100-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user_yuanjiao/private/homefile/gradscf-xc100-pbe50-20261117/logs/slurm-%j.out
#SBATCH --error=/home/user_yuanjiao/private/homefile/gradscf-xc100-pbe50-20261117/logs/slurm-%j.err

set -euo pipefail
ROOT=/home/user_yuanjiao/private/homefile/gradscf-xc100-pbe50-20261117
PYTHON=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python
cd "$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export PYTHONPATH="$ROOT/source/src"
mkdir -p "$MPLCONFIGDIR"

# Preserve a numerical traceback if the worker already wrote one. Also expose
# launch/probe failures to the controller instead of leaving a stale ready flag.
on_exit() {
    code=$?
    if test "$code" -ne 0; then
        "$PYTHON" - "$ROOT/status.json" "$code" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1])
try: status=json.loads(path.read_text())
except (OSError,ValueError): status={}
if status.get('phase')!='failed':
    status.update(phase='failed',job_id=os.environ['SLURM_JOB_ID'],time=time.time(),
                  error='Slurm launcher/worker exited with status '+sys.argv[2])
    temporary=path.with_suffix('.exit.tmp')
    temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(path)
PY
    fi
}
trap on_exit EXIT
printf '{"phase":"waiting_for_source","job_id":"%s","completed_baselines":0}\n' "$SLURM_JOB_ID" > "$ROOT/status.json.tmp"
mv "$ROOT/status.json.tmp" "$ROOT/status.json"

srun --cpu-bind=threads "$PYTHON" -u - <<'PY'
import json,os,sys,time
from pathlib import Path
import jax
import jax.numpy as jnp
affinity=sorted(os.sched_getaffinity(0))
assert 1<=len(affinity)<=int(os.environ['SLURM_CPUS_PER_TASK'])<=64
devices=jax.devices('gpu')
assert len(devices)==1 and 'V100' in devices[0].device_kind
point=jnp.eye(8,dtype=jnp.float64)
value,gradient=jax.jit(jax.value_and_grad(lambda x:jnp.sum((x@x)**2)))(point)
gradient.block_until_ready()
assert float(value)==8. and bool(jnp.all(gradient==4*point))
spectrum=jax.jit(jnp.linalg.eigvalsh)(jnp.diag(jnp.arange(1.,4.,dtype=jnp.float64)))
spectrum_error=float(jnp.max(jnp.abs(spectrum-jnp.arange(1.,4.,dtype=jnp.float64))))
spectrum_passed=bool(jnp.all(jnp.isfinite(spectrum))) and spectrum_error<=1e-12
report={'job_id':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME'],
    'python':sys.version,'jax':jax.__version__,'backend':jax.default_backend(),
    'device':devices[0].device_kind,'affinity':affinity,
    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'requested_memory':'24G','float64':bool(jax.config.jax_enable_x64),
    'gradient_passed':True,'eigenvalues':spectrum.tolist(),
    'eigenvalue_max_error':spectrum_error,'eigenvalue_tolerance':1e-12,
    'eigensolver_passed':spectrum_passed,'timestamp':time.time()}
Path('gpu_probe.json').write_text(json.dumps(report,indent=2)+'\n')
Path('allocation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
assert spectrum.dtype==jnp.float64 and spectrum_passed, report
PY

for attempt in $(seq 1 720); do
    if test -f "$ROOT/source/tools/xc100_worker.py" && test -f "$ROOT/start-worker.flag"; then
        srun --cpu-bind=threads "$PYTHON" -u "$ROOT/source/tools/xc100_worker.py" "$ROOT"
        exit 0
    fi
    if test -f "$ROOT/release.flag"; then exit 0; fi
    sleep 5
done
echo 'No verified worker/data within one hour; releasing allocation.' >&2
exit 2
