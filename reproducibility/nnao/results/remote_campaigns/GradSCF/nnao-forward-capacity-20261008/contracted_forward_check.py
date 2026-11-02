"""Small fixed-coefficient contracted-native versus primitive-projected RHF.

This CPU forward check uses no PySCF, gradients, or four-center ERI. A fixed
perturbation of the active basis-head channels stands in for frozen predictions;
it is not a MACE inference or a trained-weight validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '8')
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals, scf
from gradscf.df import build_jk_from_df
from gradscf.integrals.contraction import primitive_basis, contraction_matrix
from gradscf.integrals.density_fitting import make_auxiliary_plan, project_factors, unpack_factors
from gradscf.model.nnao import prepare_direct_basis
from gradscf.scf.core import _orthogonalizer, _diagonalize_fock, _build_density_from_occ
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--molecule', choices=('h2', 'ch4'), default='ch4')
parser.add_argument('--perturbation', type=float, default=.02)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output', type=Path,
    default=Path(__file__).with_suffix('.json'))
args = parser.parse_args()
jax.config.update('jax_enable_x64', True)
start = time.perf_counter()
if args.molecule == 'h2':
    atom = [('H', [0., 0., 0.]), ('H', [0., 0., .74])]
else:
    offsets = 1.09 / np.sqrt(3) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    atom = [('C', [0., 0., 0.])] + [('H', point) for point in offsets]
layout = prepare_direct_basis(atom, unit='Angstrom', cart=False,
    basis_family='szp663_direct', core_primitives=6)
outputs = np.array(layout.reference_outputs(), copy=True)
mask = np.zeros_like(outputs)
for owner, slot, exponents in zip(layout.shell_atoms, layout.slots, layout.parameters.exponents):
    if slot >= 0:
        mask[owner, slot, :len(exponents)] = 1.
outputs += args.perturbation * np.random.default_rng(args.seed).normal(size=outputs.shape) * mask
params = layout.bind(jnp.asarray(outputs))
top = layout.topology
primitive_top, primitive_params = primitive_basis(top, params)
aux_top, aux_params = integrals.prepare_basis(atom, 'def2-universal-jkfit', unit='Angstrom', cart=False)

def one_electron(topology, parameters):
    plan = integrals.make_plan(topology, backend='native')
    return plan.evaluate('overlap', parameters), (
        plan.evaluate('kinetic', parameters) + plan.evaluate('nuclear', parameters))

direct_s, direct_h = one_electron(top, params)
primitive_s, primitive_h = one_electron(primitive_top, primitive_params)
transform = contraction_matrix(top, params)
projected_s = transform.T @ primitive_s @ transform
projected_h = transform.T @ primitive_h @ transform
direct_aux = make_auxiliary_plan(top, aux_top)
primitive_aux = make_auxiliary_plan(primitive_top, aux_top)
direct_metric, direct_three = direct_aux.evaluate(params, aux_params)
primitive_metric, primitive_three = primitive_aux.evaluate(primitive_params, aux_params)
direct_packed = direct_aux.factors(params, aux_params)
primitive_packed = primitive_aux.factors(primitive_params, aux_params)
direct_factors = unpack_factors(direct_packed, top.nao)
projected_factors = project_factors(primitive_packed, transform)

errors = {}
def compare(name, actual, expected, *, atol=2e-9, rtol=2e-9):
    actual = np.asarray(actual); expected = np.asarray(expected)
    errors[name] = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

compare('overlap_max_abs', direct_s, projected_s, atol=2e-12)
compare('hcore_max_abs_hartree', direct_h, projected_h, atol=2e-11)
compare('auxiliary_metric_max_abs', direct_metric, primitive_metric, atol=2e-12)
compare('three_center_max_abs', unpack_factors(direct_three, top.nao),
        project_factors(primitive_three, transform), atol=2e-10)
compare('whitened_df_factors_max_abs', direct_factors, projected_factors)
occ = jnp.zeros(top.nao).at[:layout.nelectron // 2].set(2.)
_, coeff = _diagonalize_fock(direct_h, _orthogonalizer(direct_s, 1e-10))
density = _build_density_from_occ(coeff, occ)
for name, direct, projected in zip(('J_max_abs', 'K_max_abs'),
        build_jk_from_df(direct_factors, density), build_jk_from_df(projected_factors, density)):
    compare(name, direct, projected)
enuc = scf.nuclear_repulsion_energy(params.nuclear_coords, jnp.asarray(top.nuclear_charges))
config = RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
    conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
def solve(s, h, factors):
    return run_rks_from_integrals(overlap=s, hcore=h, eri=None, df_factors=factors,
        nelectron=layout.nelectron, nuclear_repulsion=enuc, ao=jnp.zeros((0, top.nao)),
        ao_deriv1=jnp.zeros((4, 0, top.nao)), grid_weights=jnp.zeros(0), config=config)
direct_result = solve(direct_s, direct_h, direct_factors)
projected_result = solve(projected_s, projected_h, projected_factors)
assert direct_result.converged and projected_result.converged
compare('scf_energy_abs_hartree', direct_result.total_energy, projected_result.total_energy, atol=1e-9, rtol=0.)
compare('scf_density_max_abs', direct_result.density_matrix, projected_result.density_matrix, atol=1e-8)
report = dict(molecule=args.molecule, basis='szp663_direct', core_primitives=6,
    cartesian=False, auxbasis='def2-universal-jkfit', nelectron=layout.nelectron,
    nao=top.nao, primitive_nao=primitive_top.nao, auxiliary_nao=aux_top.nao,
    fixed_coefficient_source='Reference basis-head outputs plus seeded active-channel perturbation; MACE not run.',
    seed=args.seed, perturbation=args.perturbation, errors=errors,
    direct_contracted_energy_hartree=direct_result.total_energy,
    primitive_projected_energy_hartree=projected_result.total_energy,
    direct_cycles=direct_result.cycles, projected_cycles=projected_result.cycles,
    contracted_packed_shape=list(direct_packed.shape), primitive_packed_shape=list(primitive_packed.shape),
    contracted_packed_bytes=int(direct_packed.size * direct_packed.dtype.itemsize),
    primitive_packed_bytes=int(primitive_packed.size * primitive_packed.dtype.itemsize),
    devices=[str(device) for device in jax.devices()], dtype=str(direct_packed.dtype),
    elapsed_seconds=time.perf_counter() - start,
    performed_gradients=False, requested_four_center_eri=False, pyscf_called=False,
    passed=True, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, indent=2), flush=True)
