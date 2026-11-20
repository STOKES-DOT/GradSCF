"""The timing entry point must preserve custom_xc energy and potential AD."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import basis_from_molecule_spec
from gradscf.data.molecule import parse_molecule_spec
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.model.autoxc import Candidate, Parameter
from gradscf.model.autoxc.workflows.xc100.molecule import GridContext, XC100Case


@pytest.fixture
def small_case():
    basis = basis_from_molecule_spec(parse_molecule_spec('H 0 0 0; H 0 0 1.4', unit='Bohr'),
                                    basis='sto-3g')
    coords = jnp.asarray([[.2, .1, .3], [-.3, .2, .8], [.5, -.1, 1.2],
                          [.1, .4, -.2], [-.2, -.3, 1.7]])
    grid = AOGrid.from_basis(basis, coords, jnp.asarray([.2, .3, .4, .2, .1]), block_size=3)
    case = object.__new__(XC100Case)
    case.floor, case.max_parameters = 1e-12, 4
    case.grid = grid
    case.transform = jnp.eye(2)
    case.context = GridContext(grid, jnp.asarray([[0., 0., 0.], [0., 0., 1.4]]), jnp.ones(2))
    case.target = SimpleNamespace(density=jnp.asarray([[1.1, .15], [.15, .9]]))
    return case


def test_energy_matrix_matches_an_independent_density_finite_difference(small_case):
    candidate = Candidate('n**1.3333333333333333*(a*u+b*z)',
        parameters=(Parameter('a', 0., -.2, .2), Parameter('b', 0., -.2, .2)), kind='energy')
    params = {'a': jnp.asarray(.01), 'b': jnp.asarray(-.02)}
    matrix = small_case.candidate_matrix(candidate)(params)
    functional = candidate.functional(lambda p: small_case.inputs(p, candidate.features))
    direction = jnp.asarray([[.3, .2], [.2, -.1]])
    energy = lambda density: small_case._energy(functional, params, density)
    density = small_case.target.density
    step = 1e-4
    finite_difference = (energy(density+step*direction)-energy(density-step*direction))/(2*step)
    np.testing.assert_allclose(jnp.sum(matrix*direction), finite_difference, atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_matrix_callable_keeps_the_affine_parameter_expansion(small_case):
    candidate = Candidate('a*n**1.3333333333333333*u',
                          parameters=(Parameter('a', 0., -.2, .2),), kind='energy')
    matrix = small_case.candidate_matrix(candidate)
    zero = {'a': jnp.asarray(0.)}
    direction = jax.jacfwd(matrix)(zero)['a']
    np.testing.assert_allclose(matrix(zero), 0., atol=1e-14)
    np.testing.assert_allclose(matrix({'a': jnp.asarray(.017)}), .017*direction, atol=1e-12)


def test_potential_matrix_matches_direct_quadrature(small_case):
    candidate = Candidate('a*n', parameters=(Parameter('a', 0., -.2, .2),), kind='potential')
    matrix = small_case.candidate_matrix(candidate)({'a': jnp.asarray(.03)})
    ao, _, _, _, weights = small_case.grid.block(0, 5)
    rho = jnp.einsum('gi,ij,gj->g', ao, small_case.target.density, ao)
    field = .03*jnp.sqrt(rho*rho+small_case.floor**2)
    expected = jnp.einsum('g,gi,gj->ij', weights*field, ao, ao)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_matrix_callable_rejects_an_undeclared_feature(small_case):
    with pytest.raises(ValueError, match='declared density features'):
        small_case.candidate_matrix(Candidate('a*missing', parameters=(Parameter('a', 0., -.2, .2),)))


def test_full_energy_reproduces_and_replaces_pbe(small_case):
    from gradscf.dft.iks import Target

    case = small_case
    case.system_id, case.timings, case.max_parameters = 'toy', {}, None
    case.target = Target(case.target.density, jnp.eye(2), 2)
    case.arrays = {'kinetic': jnp.eye(2), 'nuclear': jnp.diag(jnp.asarray([0., 2.])),
                   'hartree': jnp.zeros((2, 2))}
    case.prepare(include_energy=False)
    pbe = case.bind(Candidate('pbe_x + pbe_c', kind='energy'))['toy']
    zero = case.bind(Candidate('0*n', kind='energy'))['toy']
    np.testing.assert_allclose(pbe({}).reward, 0., atol=1e-10)
    np.testing.assert_allclose(zero({}).fock, case.problem.base_fock, atol=1e-12)
    assert case.baseline_energy is None and bool(pbe({}).valid) and bool(zero({}).valid)

    edited = case.bind(Candidate('pbe_x * exp(a*u) + pbe_c',
                                 parameters=(Parameter('a', 0., -.5, .5),),
                                 kind='energy'))['toy']
    score = lambda a: edited({'a': a}).reward
    step = 1e-4
    np.testing.assert_allclose(jax.grad(score)(.1), (score(.1+step)-score(.1-step))/(2*step),
                               atol=1e-8, rtol=1e-5)
