#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONPATH=src
py=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
work=artifacts/aniline-basis-comparison-20260920
"$py" - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('artifacts/aniline-basis-comparison-20260920/source-hashes.json').read_text())
bad=[p for p,h in m.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not bad,bad
import gradscf.model.nnao,gradscf.integrals
print('Verified source files:',len(m),flush=True)
print('NNAO:',gradscf.model.nnao.__file__,flush=True)
print('Integrals:',gradscf.integrals.__file__,flush=True)
PY
taskset -c 0-3 "$py" "$work/runner.py" test -q tests/nnao/test_molecular_benchmark.py tests/nnao/test_core6.py tests/nnao/test_szp663.py > "$work/remote-tests.log" 2>&1 || { cat "$work/remote-tests.log"; exit 1; }
cat "$work/remote-tests.log"
taskset -c 0-3 "$py" tools/compare_methane_core6_basis_sets.py "$work/comparison-input.json" --output "$work/remote-smoke" --bases sto-3g 3-21g 6-31g > "$work/remote-smoke.log" 2>&1 || { cat "$work/remote-smoke.log"; exit 1; }
taskset -c 0-3 "$py" tests/comparisons/validate_methane_core6_basis_sets.py "$work/remote-smoke/comparison.json" "$work/comparison-input.json" > "$work/remote-reference.log" 2>&1
cat "$work/remote-smoke.log"
cat "$work/remote-reference.log"
