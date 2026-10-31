"""H2/PBE0 exact-pair scGW: production versus dense implicit response, no FD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from gradscf import dft, gto, gw
from gradscf.gw.g0w0 import _mo_factors
from gradscf.gw.scgw_response import _pack_state, _residual
from gradscf_tools.gw_response import MolecularGW

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = perf_counter()
mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
mf = dft.RKS(mol, xc="pbe0", grids_level=0, conv_tol=1e-12,
             conv_tol_grad=1e-10, max_cycle=100).run()
assert mf.converged
model = MolecularGW(mf, method="scgw", nw=24, max_cycle=600, beta=20., mixing=.5)
coordinates = model.coordinates
direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
seed_seconds = perf_counter() - started


def packed_and_params(r):
    common, valid = model._prepare(r)
    result = gw.scgw_matsubara_restricted(**common, beta=model.beta, mixing=model.mixing,
        max_iter=model.max_cycle, tol=1e-10, particle_tol=1e-12,
        differentiation=model.outer_response)
    c = common["mo_coeff"]
    params = dict(hcore=c.T @ common["hcore_matrix"] @ c,
                  b=_mo_factors(common["df_factors"], c), phonons=None)
    state = _pack_state(result.fock_mo, result.self_energy_iw, result.sigma_moment,
                        result.chemical_potential, result.grid)
    aux = (result.grid, valid, result.converged, result.fixed_point_residual,
           result.particle_number_error)
    return (state, params), aux


print("start joint preparation/production JVP", flush=True)
started = perf_counter()
(state, params), (production, dparams), aux = jax.jit(lambda r: jax.jvp(
    packed_and_params, (r,), (direction,), has_aux=True))(coordinates)
jax.block_until_ready((state, production, dparams, aux))
joint_seconds = perf_counter() - started
grid, reference_valid, converged, fixed_error, particle_error = aux


def dense_system(state, params, dparams):
    residual = lambda s, p: _residual(s, p, grid, model.nocc)
    matrix = jax.jacfwd(lambda s: residual(s, params))(state)
    forcing = jax.jvp(lambda p: residual(state, p), (params,), (dparams,))[1]
    return matrix, forcing, residual(state, params)


print("start dense residual Jacobian/forcing", flush=True)
started = perf_counter()
matrix, forcing, residual = jax.jit(dense_system)(state, params, dparams)
jax.block_until_ready((matrix, forcing, residual))
dense_seconds = perf_counter() - started
matrix, forcing, residual, production = map(np.asarray, (matrix, forcing, residual, production))
dense = np.linalg.solve(matrix, -forcing)
primal_correction = np.linalg.solve(matrix, -residual)
singular = np.linalg.svd(matrix, compute_uv=False)
schur = matrix[-1, -1] - matrix[-1, :-1] @ np.linalg.solve(matrix[:-1, :-1], matrix[:-1, -1])
norms = lambda value: dict(l2=float(np.linalg.norm(value)), maxabs=float(np.max(np.abs(value))))
payload = dict(system="H2/STO-3G Cartesian PBE0 grid0", units="Hartree/bohr",
    beta=model.beta, nw=model.nw, max_cycle=model.max_cycle, state_dimension=len(state),
    devices=[str(device) for device in jax.devices()], affinity=sorted(os.sched_getaffinity(0)),
    seed_seconds=seed_seconds, joint_jvp_seconds=joint_seconds, dense_jacobian_seconds=dense_seconds,
    reference_valid=bool(reference_valid), converged=bool(converged),
    fixed_point_residual=float(fixed_error), particle_number_error=float(particle_error),
    full_residual=norms(residual), cond_a=float(singular[0] / singular[-1]),
    min_singular_value=float(singular[-1]), schur=float(schur), beta_schur=float(model.beta * schur),
    production_mu_jvp=float(production[-1]), dense_mu_jvp=float(dense[-1]),
    primal_newton_mu_correction=float(primal_correction[-1]),
    mu_jvp_difference=float(production[-1] - dense[-1]), tangent_difference=norms(production - dense),
    production_linear_residual=norms(matrix @ production + forcing),
    dense_linear_residual=norms(matrix @ dense + forcing), forcing=norms(forcing))
source = Path(_residual.__code__.co_filename).resolve()
payload["response_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
