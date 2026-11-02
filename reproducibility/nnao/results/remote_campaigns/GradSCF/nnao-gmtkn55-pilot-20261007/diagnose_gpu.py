"""Observe the existing implicit linear solves without changing their results."""
import importlib
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import verify_source_imports
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['JAX_ENABLE_X64'] = '1'
import jax
jax.config.update('jax_enable_x64', True)
verify_source_imports(ROOT)
from optimize_methane_nnao import MethaneRHF as experiment_class
import jax.numpy as jnp

module = importlib.import_module('gradscf.solvers.linear.gmres')
original = module.gmres


def observed(operator, rhs, **kwargs):
    result, info = original(operator, rhs, **kwargs)
    applied = operator(result)
    jax.debug.print('method={method} residual={residual} rhs_norm={rhs_norm} '
                    'finite_solution={finite_solution} finite_operator={finite_operator}',
        method=kwargs['solve_method'], residual=jnp.linalg.norm(applied-rhs),
        rhs_norm=jnp.linalg.norm(rhs), finite_solution=jnp.all(jnp.isfinite(result)),
        finite_operator=jnp.all(jnp.isfinite(applied)))
    return result, info


module.gmres = observed
linear = importlib.import_module('gradscf.solvers.linear.implicit')
wrapped = linear.gmres_solve


def inspect_final(operator, rhs, **controls):
    solution = wrapped(operator, rhs, **controls)
    residual = rhs - operator(solution)
    threshold = controls['atol'] + controls['rtol'] * jnp.linalg.norm(rhs)
    jax.debug.print('FINAL before correction residual={r} threshold={t} rhs={b}',
        r=jnp.linalg.norm(residual), t=threshold, b=jnp.linalg.norm(rhs))
    correction = wrapped(operator, residual, **controls)
    candidate = solution + correction
    candidate_residual = rhs - operator(candidate)
    jax.debug.print('CORRECTED residual={r} threshold={t}',
        r=jnp.linalg.norm(candidate_residual), t=threshold)
    # Diagnostic only: observe one numerical residual correction in the same
    # opaque solve, without changing the differentiated response operator.
    improve = jnp.linalg.norm(candidate_residual) < jnp.linalg.norm(residual)
    return jnp.where(improve, candidate, solution)


linear.gmres_solve = inspect_final
data, cache = map(Path, sys.argv[1:])
row = next(json.loads(line) for line in data.read_text().splitlines()
           if json.loads(line)['structure_id'] == 'S22/08')
row['name'] = row['structure_id']
experiment = experiment_class(geometry=row, basis_family='szp663_direct',
    core_primitives=6, jk_backend='df', integral_cache=cache)
(energy, diagnostics), gradient = experiment._value_grad(
    experiment.layout.reference_outputs(), experiment.ps, experiment.ph, experiment.rep)
print('energy', float(energy), flush=True)
print('primal', {key: float(value) for key, value in diagnostics.items()}, flush=True)
print('finite_gradient', bool(jnp.all(jnp.isfinite(gradient))), flush=True)
