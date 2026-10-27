"""Contraction-parameter response through the shared SCF backward boundary.

This bounded H2 fixture keeps geometry/exponents fixed and projects libcint
primitive values with the existing normalized contraction map. It preserves
coefficient response through both SCF modes without a large-system ERI cache.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.scf import nuclear_repulsion_energy
from gradscf.scf.orbital_optimization import minimize_uks_from_integrals


@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_h2_321g_contraction_backward_matches_finite_difference(mode):
    from scipy.linalg import eigh
    from gradscf._memory import workspace_bytes
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix, contract_integrals
    topology, parameters = integrals.prepare_basis('H 0 0 0; H 0 0 .74',basis='3-21g')
    primitive_topology, primitive_parameters = primitive_basis(topology, parameters)
    n, m = primitive_topology.nao, topology.nao
    required = 8*(n**4+4*n**2+n**3*m+n**2*m**2+n*m**3+m**4)
    assert required <= workspace_bytes(64., device='cpu')
    plan = integrals.make_plan(primitive_topology)
    tensors = {operator: plan.evaluate(operator, primitive_parameters)
               for operator in ('overlap', 'kinetic', 'nuclear', 'eri')}
    def physical(p):
        return contract_integrals(tensors, contraction_matrix(topology, p))
    initial = physical(parameters)
    s = initial['overlap']
    h = initial['kinetic']+initial['nuclear']
    _, coeff = eigh(np.asarray(h),np.asarray(s))
    seed = np.stack([coeff,coeff]); occ = np.zeros((2,len(coeff)));occ[:,0]=1.
    enuc = nuclear_repulsion_energy(parameters.nuclear_coords,jnp.asarray(topology.nuclear_charges))

    def calculate(theta):
        first = parameters.coefficients[0].at[0,0].set(theta)
        moving = replace(parameters,coefficients=(first,*parameters.coefficients[1:]))
        inputs = physical(moving)
        return minimize_uks_from_integrals(overlap=inputs['overlap'],
            hcore=inputs['kinetic']+inputs['nuclear'],
            eri=inputs['eri'],nuclear_repulsion=enuc,
            ao=jnp.zeros((0,len(coeff))),ao_deriv1=jnp.zeros((4,0,len(coeff))),grid_weights=jnp.zeros(0),
            mo_coeff=seed,mo_occ=occ,max_iterations=100,gradient_tolerance=1e-9,
            gradient_mode=mode,orthonormalize_initial=True)

    def observable(theta):
        result = calculate(theta)
        return result.total_energy+.1*result.density_matrix[0,0,1]

    theta, step = float(parameters.coefficients[0][0,0]), 1e-4
    value = jax.jit(observable)
    actual = jax.jit(jax.grad(observable))(theta)
    fd = (8*(value(theta+step)-value(theta-step))-value(theta+2*step)+value(theta-2*step))/(12*step)
    assert calculate(theta).stationary
    np.testing.assert_allclose(actual,fd,atol=5e-7,rtol=0)
