"""CH4/def2-SVP, grid1: cached CCSD/Wu-Yang reference -> custom_xc reward.

Set GRADSCF_IKS_TEACHER to the qualified teacher JSON. The matching
.reference.npz can be relocated with GRADSCF_IKS_REFERENCE. Raw data stay
outside Git. No HF, candidate SCF or new inversion is run by this example.
The existing iKS loader checks data hashes, AO conventions and teacher flags.
Run from the repository root; no CLI. Atomic units and float64 throughout.
"""
from pathlib import Path
from time import perf_counter
import json
import os
import platform

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import dft
from gradscf.dft import iks
from gradscf_tools.pbe_loop import load_teacher_reference, build_single_pbe_input


started = perf_counter()
teacher_file = Path(os.environ.get('GRADSCF_IKS_TEACHER', 'iks_data/teacher.json'))
reference_file = Path(os.environ.get('GRADSCF_IKS_REFERENCE',
                                     str(teacher_file.with_suffix('.reference.npz'))))
teacher = load_teacher_reference(teacher_file, reference_npz=reference_file)
state, nuclear_repulsion = build_single_pbe_input(teacher, max_memory_mb=4096.)
reference, grid = teacher.reference, state.grid_ao
preparation_seconds = perf_counter()-started
reference_density_tolerance = 1e-3  # Relative grid L2; this sample's quality gate.


# An external analytic functional: E_x = -a C_x int(n_alpha^(4/3)+n_beta^(4/3)).
# The same custom_xc interface also accepts an externally constructed network.
exchange = dft.custom_xc(
    input_fn=lambda fields: (fields.rho_spin, fields.weights),
    energy_fn=lambda a, inputs: -a*.75*(6/jnp.pi)**(1/3)*jnp.sum(
        inputs[1]*jnp.sum(jnp.maximum(inputs[0], 1e-18)**(4/3), axis=0)),
    name='scaled_Dirac_exchange')
energy = iks.energy_function(state, exchange, layout='rks',
    nuclear_repulsion=nuclear_repulsion)


def assess(a):
    return iks.reward(reference, energy, a, grid=grid,
        energy_scale=.01, density_scale=.01, weights={'gap': 1., 'density': 1.})


compiled = jax.jit(assess)
# Record compilation separately from warm evaluation; synchronize JAX results.
start = perf_counter()
compiled(1.).value.block_until_ready()
compile_seconds = perf_counter()-start
rows = []
for a in (.8, 1., 1.2):
    start = perf_counter()
    result = compiled(a)
    result.value.block_until_ready()
    rows.append(dict(formula='-a * Cx * integral(sum_sigma n_sigma**(4/3))', a=a,
        reward=float(result.value), loss=float(result.loss), valid=bool(result.valid),
        variational_gap_hartree=float(result.variational_gap),
        signed_wu_yang_gap_hartree=float(result.wu_yang_gap),
        one_shot_density_relative_l2=float(result.density_error),
        reference_density_relative_l2=float(result.reference_density_error),
        frozen_reference_energy_hartree=float(result.total_energy),
        occupied_virtual_gap_hartree=float(result.gap), seconds=perf_counter()-start))
    print(rows[-1], flush=True)

value, derivative = jax.jit(jax.value_and_grad(lambda a: assess(a).value))(1.)
step = 1e-4
finite_difference = (compiled(1.+step).value-compiled(1.-step).value)/(2*step)
np.testing.assert_allclose(derivative, finite_difference, rtol=1e-5, atol=1e-8)
report = dict(reference='CAFES-QM9 unrelaxed CCSD Lambda 1-RDM',
    atoms=np.asarray(teacher.atomic_numbers).tolist(),
    positions_angstrom=np.asarray(teacher.positions_angstrom).tolist(),
    basis=teacher.provenance['inversion_basis'], grid_level=teacher.provenance['grid_level'],
    grid_points=int(grid.ngrids), nao=int(teacher.overlap.shape[0]),
    reference_energy_hartree=float(teacher.target.energy),
    inverse_method=reference.method, reference_provenance=teacher.provenance,
    reference_density_tolerance=reference_density_tolerance,
    preparation_seconds=preparation_seconds, compile_seconds=compile_seconds,
    energy_scale_hartree=.01, density_scale_relative_l2=.01,
    weights={'gap': 1., 'density': 1.},
    backend=jax.default_backend(), dtype='float64', jax=jax.__version__,
    python=platform.python_version(), hardware=platform.processor(),
    command='PYTHONPATH=src JAX_PLATFORMS=cpu python examples/iks/single_sample_reward.py',
    reward_ad=float(derivative), reward_finite_difference=float(finite_difference), candidates=rows)
folder = Path(os.environ.get('GRADSCF_IKS_OUTPUT', 'artifacts/iks-single-reward'))
folder.mkdir(parents=True, exist_ok=True)
(folder/'reward.json').write_text(json.dumps(report, indent=2)+'\n')
if not all(row['valid'] and row['reference_density_relative_l2'] <= reference_density_tolerance
           for row in rows):
    raise RuntimeError('Candidate spectral checks or reference density qualification failed')

# jy_gpu_2/n002, CPU4, JAX0.10.0, float64; qualified cached CH4 reference:
# a       reward       one-shot density relative L2   variational gap / Ha
# 0.8     0.05656856    0.03221823                    0.06297521
# 1.0     0.14198973    0.01592318                    0.03507287
# 1.2     0.13533647    0.01582798                    0.03883741
# Reference inverse density relative L2: 0.0007242068509226423
# Reward parameter derivative, AD / FD(h=1e-4): 0.48912303168 / 0.48912290713
# These are frozen-reference/one-shot diagnostics, not converged SCF errors.
