"""Compare CUDA response with/without a registered host CPU backend."""
import json
import os
from pathlib import Path
import sys

profile, destination = sys.argv[1:]
os.environ['JAX_PLATFORMS'] = profile
os.environ['JAX_ENABLE_X64'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, verify_source_imports
from optimize_methane_nnao import MethaneRHF
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
data = Path('/home/user_yuanjiao/private/datafile/GradSCF_GMTKN55/selections/hf_pilot_20261007/structures.jsonl')
row = next(json.loads(line) for line in data.read_text().splitlines()
           if json.loads(line)['structure_id'] == 'S22/08')
row['name'] = row['structure_id']
cache = ROOT / 'artifacts/nnao-gmtkn55-pilot-20261007/job-93084/cache/ec70645f188a04a1/integrals.npz'
ex = MethaneRHF(geometry=row, basis_family='szp663_direct', core_primitives=6,
               jk_backend='df', integral_cache=cache)
(energy, info), gradient = ex._value_grad(ex.layout.reference_outputs(), ex.ps, ex.ph, ex.rep)
report = dict(profile=profile, default_backend=jax.default_backend(),
    imported_sources=verify_source_imports(ROOT),
    energy_device=str(energy.device), gradient_device=str(gradient.device),
    integral_device=str(ex.rep.device), energy_hartree=float(energy),
    gradient_finite=bool(jnp.isfinite(gradient).all()),
    nonfinite_gradient_count=int(jnp.sum(~jnp.isfinite(gradient))),
    diagnostics={key: float(value) for key, value in info.items()})
atomic_json(destination, report)
print(json.dumps(report), flush=True)
