#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=anthracene-AD-restart
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:NV5090:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
RUN="$SLURM_SUBMIT_DIR"
PY=/home/user_yuanjiao/private/homefile/gradscf-libcint-nnao-20261026/env-compare/bin/python
cd "$RUN"
export RUN PY PYTHONPATH="$RUN/src:$RUN:$RUN/tests"
export JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
scontrol show job "$SLURM_JOB_ID" > results/allocation.txt
lscpu > results/lscpu.txt
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv > results/gpu.csv
sha256sum src/gradscf/integrals/_native/lib/libgradscf_integrals.so > results/native.sha256
run_stage() {
    local stage="$1" code=0
    shift
    printf 'START %s %s\n' "$stage" "$(date -Is)"
    "$@" > "results/$stage.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "results/$stage.exit"
    printf 'END %s code=%s %s\n' "$stage" "$code" "$(date -Is)"
    return "$code"
}
export JAX_PLATFORMS=cpu GRADSCF_NATIVE_THREADS=8
# Reuse the completed native B3LYP geometry check only for identical code.
"$PY" - <<'PY'
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
checks = {'examples/tdscf/anthracene_ht/05_ad_response.py':
          '30934c24ea4133c1a2e9190b770b2893f7a1a05c132b5b6482b30125e941a603',
          'tests/test_full_tddft_geometry_ad.py':
          'a061ee1383381bb60859b94bc2001dae9b4be67db0e8550d855c14101da3c7cd'}
for name, expected in checks.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
cases = [case for case in ET.parse('results/smoke.xml').getroot().iter('testcase')
         if case.get('name', '').endswith('[b3lyp]')]
assert len(cases) == 1 and not any(cases[0].find(tag) is not None
                                 for tag in ('failure', 'error', 'skipped'))
print('Reused passed B3LYP geometry check with identical source hashes.', flush=True)
PY
run_stage "mechanism-$SLURM_JOB_ID" srun -n1 -c8 --cpu-bind=cores timeout 300 "$PY" -m pytest -q \
    tests/test_anthracene_pullback_jit.py --junitxml="results/mechanism-$SLURM_JOB_ID.xml"
export JAX_PLATFORMS=cuda,cpu GRADSCF_NATIVE_THREADS=64
export GRADSCF_ANTHRACENE_INPUT="$RUN/examples/tdscf/anthracene_ht/input_ad_eight_states.json"
export GRADSCF_ANTHRACENE_OUTPUT="$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/raw"
export GRADSCF_ANTHRACENE_FD_OUTPUT="$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw"
run_stage anthracene-ad srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard \
    --seconds 72000 --rss-bytes 236223201280 results/anthracene-ad.process.json -- \
    "$PY" examples/tdscf/anthracene_ht/05_ad_response.py
export JAX_PLATFORMS=cpu GRADSCF_NATIVE_THREADS=8
run_stage spectrum srun -n1 -c8 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard \
    --seconds 300 --rss-bytes 8589934592 results/spectrum.process.json -- \
    "$PY" examples/tdscf/anthracene_ht/04_spectrum.py
