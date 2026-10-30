"""H2/def2-SVP, grid1: PBE density -> offline Wu-Yang -> PBE assessment.

The reference is a known PBE solution, not a high-level correlated teacher.
Finite-basis density reconstruction does not certify a unique real-space
potential. No CLI; run from the repository root. Atomic units and float64.
"""
from pathlib import Path
from time import perf_counter
import json
import os

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from gradscf import dft, gto, integrals
from gradscf.dft import iks
from gradscf.integrals.grids.blocks import AOGrid


started = perf_counter()
mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='def2-svp', cart=True)
mf = dft.RKS(mol, xc='pbe', grids_level=1, execution_device='cpu',
             max_memory=2048., conv_tol=1e-11, conv_tol_grad=1e-9).run()
if not mf.converged:
    raise RuntimeError('The known PBE reference did not converge')
state = mf.to_reference()
basis = integrals.basis_from_molecule_spec(mol.to_spec(), basis=mol.basis)
grid = AOGrid.from_basis(basis, state.grid.coords, state.grid.weights, block_size=512)
potential = iks.Potential.ao_products(grid, state.overlap_matrix, max_memory=2048.)
target = iks.Target(mf.make_rdm1(), state.overlap_matrix, mol.nelectron, energy=mf.e_tot)

# The base is kinetic plus the LOCAL nuclear potential. A frozen HF Fock
# would retain nonlocal exchange and would define a different inverse problem.
result = iks.invert(target, potential, state.h1e, method='wu_yang',
                    regularization=1e-12, tolerance=1e-6,
                    gradient_tolerance=1e-9, maxiter=500)
report = dict(reference='PBE known-density round trip', molecule='H2',
    basis='def2-SVP Cartesian', grid_level=1, grid_points=int(grid.ngrids),
    scf_converged=bool(mf.converged), scf_total_energy_hartree=float(mf.e_tot),
    inverse_method=result.method, valid=bool(result.valid), converged=bool(result.converged),
    density_error_l2=float(result.density_error), moment_error=float(result.moment_error),
    raw_moment_error=float(result.raw_moment_error), gap_hartree=float(result.gap),
    projection_rank=int(result.projection_rank), iterations=int(result.iterations),
    gradient_norm=float(result.gradient_norm), optimizer_method=result.optimizer_method, optimizer_evaluations=result.optimizer_evaluations,
    refinement_evaluations=result.refinement_evaluations, elapsed_seconds=perf_counter()-started,
    backend=jax.default_backend(), dtype='float64', jax=jax.__version__)
if result.valid:
    energy = iks.energy_function(state, 'pbe', layout='rks',
                                 nuclear_repulsion=mol.to_spec().nuclear_repulsion)
    score = iks.evaluate(result.reference, energy, quantities=('energy', 'stationarity'))
    report.update(pbe_energy_error_hartree=float(score.energy_error),
                  pbe_rotation_norm2=float(score.rotation_norm2), evaluation_valid=bool(score.valid))
folder = Path(os.environ.get('GRADSCF_IKS_OUTPUT', 'artifacts/iks-wu-yang-pbe'))
folder.mkdir(parents=True, exist_ok=True)
(folder/'roundtrip.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
if not result.valid:
    raise RuntimeError('Wu-Yang density/spectral validation failed; no teacher was produced')

# jy_gpu_2/n005, Slurm110759, CPU4/8GiB, JAX0.10.0, float64:
# PBE total energy: -1.1600208891190564 Ha
# Inverse valid/converged: True / True
# Density L2 error: 7.515922384873537e-14 bohr^-3/2
# Normalized moment error: 3.2142194961661105e-13
# Gradient max: 3.2978119925237284e-16; gap: 0.6490452061649511 Ha
# PBE energy reassessment error: -4.884981308350689e-15 Ha
# PBE rotation residual squared: 5.191049227425028e-21
# Cold preparation/inversion elapsed: 16.544858856126666 s
# This is a known-functional round trip, not a high-level density benchmark.
