"""H2/PBE0 exact-pair scGW forward caps; no derivatives or finite differences."""
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
from gradscf import dft, gto
from gradscf.gw.g0w0 import _mo_factors
from gradscf.gw.matsubara import matsubara_grid
from gradscf.gw.scgw_response import _solve_state
from gradscf_tools.gw_response import MolecularGW

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = perf_counter()
mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
mf = dft.RKS(mol, xc="pbe0", grids_level=0, conv_tol=1e-12,
             conv_tol_grad=1e-10, max_cycle=100).run()
assert mf.converged
model = MolecularGW(mf, method="scgw", nw=24, beta=20., mixing=.5, conv_tol=1e-12)
common, valid = model._prepare(model.coordinates)
jax.block_until_ready((common, valid))
assert bool(valid)
c = common["mo_coeff"]
h = c.T @ common["hcore_matrix"] @ c
b = _mo_factors(common["df_factors"], c)
h, b = .5 * (h + h.T), .5 * (b + b.swapaxes(-1, -2))
grid = matsubara_grid(nw=model.nw, beta=model.beta)
prepare_seconds = perf_counter() - started
run = jax.jit(lambda cap: _solve_state.__wrapped__(h, b, common["mo_energy"], grid,
    nocc=model.nocc, max_iter=cap, tol=model.conv_tol, mixing=model.mixing, particle_tol=1e-12))
started = perf_counter()
compiled = run.lower(jnp.asarray(200, dtype=jnp.int32)).compile()
compile_seconds = perf_counter() - started
records = []
for cap in (200, 600, 1600):
    started = perf_counter()
    state, iterations, residual, valid = compiled(jnp.asarray(cap, dtype=jnp.int32))
    jax.block_until_ready((state, iterations, residual, valid))
    records.append(dict(cap=cap, iterations=int(iterations), residual=float(residual),
                        valid=bool(valid), mu=float(state[-1]), seconds=perf_counter() - started))
    print(json.dumps(records[-1]), flush=True)
source = Path(_solve_state.__wrapped__.__code__.co_filename).resolve()
payload = dict(system="H2/STO-3G Cartesian PBE0 grid0", units="Hartree",
    beta=model.beta, nw=model.nw, tol=model.conv_tol, particle_tol=1e-12, mixing=model.mixing,
    affinity=sorted(os.sched_getaffinity(0)), prepare_seconds=prepare_seconds,
    compile_seconds=compile_seconds, response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    records=records)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n")
