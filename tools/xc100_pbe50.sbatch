#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=gradscf-xc100-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:5090:1
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yjiao/gradscf_checks/xc100-pbe-50-20261116/logs/slurm-%j.out
#SBATCH --error=/home/yjiao/gradscf_checks/xc100-pbe-50-20261116/logs/slurm-%j.err

set -euo pipefail
ROOT=/home/yjiao/gradscf_checks/xc100-pbe-50-20261116
PYTHON=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
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
printf '{"phase":"waiting_for_source","job_id":"%s","completed_baselines":0}\n' "$SLURM_JOB_ID" > "$ROOT/status.json.tmp"
mv "$ROOT/status.json.tmp" "$ROOT/status.json"

# Slurm CPU IDs are topology-ordered, not Linux processor numbers. This node
# has no task/affinity plugin, so bind only this job's children explicitly.
CPUS=$(/usr/bin/python3 - <<'PY'
import json,os,re,subprocess
job=subprocess.check_output(['scontrol','show','job','-d',os.environ['SLURM_JOB_ID']],universal_newlines=True)
match=re.search(r'CPU_IDs=([^\s]+)',job)
if match is None: raise RuntimeError('No allocated CPU list')
ids=[]
for group in match.group(1).split(','):
    parts=group.split('-'); ids.extend(range(int(parts[0]),int(parts[-1])+1))
topology=[]
for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],universal_newlines=True).splitlines():
    if not line.startswith('#'):
        cpu,core,socket=map(int,line.split(','));topology.append((socket,core,cpu))
ordered=[row[2] for row in sorted(topology)]
cpus=sorted(ordered[i] for i in ids)
if len(cpus)!=64 or len(set(cpus))!=64: raise RuntimeError('Expected exactly 64 allocated CPUs')
with open('allocation.json','w') as handle:
    json.dump({'job_id':os.environ['SLURM_JOB_ID'],'abstract_cpu_ids':ids,'linux_cpu_ids':cpus,
               'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'requested_memory':'512G',
               'note':'Node uses CR_CORE; requested memory is not a scheduler-enforced RAM reservation.'},handle,indent=2)
print(','.join(map(str,cpus)))
PY
)

taskset -c "$CPUS" "$PYTHON" -u - <<'PY'
import json,os,sys,time
from pathlib import Path
import jax
import jax.numpy as jnp
assert len(os.sched_getaffinity(0))==64
devices=jax.devices('gpu')
assert len(devices)==1 and '5090' in devices[0].device_kind
point=jnp.eye(8,dtype=jnp.float64)
value,gradient=jax.jit(jax.value_and_grad(lambda x:jnp.sum((x@x)**2)))(point)
gradient.block_until_ready()
assert float(value)==8. and bool(jnp.all(gradient==4*point))
report={'job_id':os.environ['SLURM_JOB_ID'],'python':sys.version,'jax':jax.__version__,
        'backend':jax.default_backend(),'device':devices[0].device_kind,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'affinity':sorted(os.sched_getaffinity(0)),'float64':bool(jax.config.jax_enable_x64),
        'matrix_value':float(value),'gradient_passed':True,'timestamp':time.time()}
Path('gpu_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
PY

# Give source/data preparation one bounded hour; do not hold an unused card
# indefinitely if the controller disappears. The worker owns later idle limits.
for attempt in $(seq 1 720); do
    if test -f "$ROOT/source/tools/xc100_worker.py" && test -f "$ROOT/start-worker.flag"; then
        exec taskset -c "$CPUS" "$PYTHON" -u "$ROOT/source/tools/xc100_worker.py" "$ROOT"
    fi
    if test -f "$ROOT/release.flag"; then exit 0; fi
    sleep 5
done
echo 'No prepared worker within one hour; releasing allocation.' >&2
exit 2
