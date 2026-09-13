"""Contraction-parameter response through the shared SCF backward boundary.

Use the explicit JAX reference integral backend: native contraction derivatives
are not implemented and must not be silently substituted by this test.
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
    topology, parameters = integrals.prepare_basis('H 0 0 0; H 0 0 .74',basis='3-21g')
    plan = integrals.make_plan(topology,backend='jax_reference')
    s = plan.evaluate('overlap',parameters)
    h = plan.evaluate('kinetic',parameters)+plan.evaluate('nuclear',parameters)
    _, coeff = eigh(np.asarray(h),np.asarray(s))
    seed = np.stack([coeff,coeff]); occ = np.zeros((2,len(coeff)));occ[:,0]=1.
    enuc = nuclear_repulsion_energy(parameters.nuclear_coords,jnp.asarray(topology.nuclear_charges))

    def calculate(theta):
        first = parameters.coefficients[0].at[0,0].set(theta)
        moving = replace(parameters,coefficients=(first,*parameters.coefficients[1:]))
        return minimize_uks_from_integrals(overlap=plan.evaluate('overlap',moving),
            hcore=plan.evaluate('kinetic',moving)+plan.evaluate('nuclear',moving),
            eri=plan.evaluate('eri',moving),nuclear_repulsion=enuc,
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
