"""Native atomic seeds and DF orbital curvature, without finite differences."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'comparisons'))
from nnao_reactivity.rescue import atomic_occupations, minao_guess, df_curvature
from nnao_core.native import make_inputs, traditional_shells


def test_neutral_carbon_and_bromine_atomic_populations_are_spherical():
    carbon = atomic_occupations(6, [(0, 2), (1, 1)])
    np.testing.assert_allclose(carbon, [2., 2., 2/3, 2/3, 2/3])
    bromine = atomic_occupations(35, [(0, 4), (1, 3), (2, 1)])
    assert len(bromine) == 18
    np.testing.assert_allclose(bromine.sum(), 35.)
    np.testing.assert_allclose(bromine[14:18], [2., 2., 2., 2.])


def test_native_minao_guess_has_correct_spin_charge_and_positive_density():
    row = dict(symbols=['H', 'H'], atomic_numbers=[1, 1], coords_angstrom=[[0., 0., 0.], [0., 0., .74]])
    shells = traditional_shells(row, '6-31g(d,p)')
    data = make_inputs(row, shells)
    density = minao_guess(row, shells, data['overlap'], [1, 1], nelectron=2, spin=0)
    np.testing.assert_allclose(np.einsum('spq,qp->s', density, data['overlap']), [1., 1.], atol=1e-12)
    assert np.linalg.eigvalsh(density).min() > -1e-12
    ghost = minao_guess(row, shells, data['overlap'], [1, 0], nelectron=1, spin=1)
    np.testing.assert_allclose(np.einsum('spq,qp->s', ghost, data['overlap']), [1., 0.], atol=1e-12)


@pytest.mark.parametrize('occupations', [((1., 0., 0.), (1., 0., 0.)),
                                        ((1., 1., 0.), (1., 0., 0.))])
def test_df_spin_coupled_hessian_matches_independent_autodiff(occupations):
    from gradscf.scf.orbital_optimization import _rotation_functions
    rng = np.random.default_rng(31)
    coeff = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(2)])
    b = rng.normal(size=(4, 3, 3))*.13
    b = jnp.asarray((b+b.transpose(0, 2, 1))/2)
    h = jnp.diag(jnp.array([-1.1, -.4, .7]))
    gradient, apply, _ = df_curvature(h, b, jnp.asarray(coeff), occupations)
    dimension, rotate, density_from = _rotation_functions('uks', occupations)
    def energy(angles):
        density = density_from(rotate(angles, jnp.asarray(coeff)))
        total = density.sum(axis=0)
        rho = jnp.einsum('Qij,ij->Q', b, total)
        exchange = sum(jnp.einsum('Qij,jk,Qkl,li->', b, dm, b, dm) for dm in density)
        return jnp.sum(total*h)+.5*jnp.sum(rho*rho)-.5*exchange
    origin = jnp.zeros(dimension)
    reference_gradient, reference_hvp = jax.linearize(jax.grad(energy), origin)
    vector = jnp.asarray(rng.normal(size=dimension))
    np.testing.assert_allclose(gradient, reference_gradient, atol=1e-11)
    np.testing.assert_allclose(apply(vector), reference_hvp(vector), atol=1e-11)


def test_zero_orbital_gradient_does_not_make_an_excited_root_stable():
    h = jnp.diag(jnp.array([-1., .2]))
    c = jnp.array([[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]])
    gradient, apply, _ = df_curvature(h, jnp.zeros((1, 2, 2)), c, ((1., 0.), (0., 0.)))
    np.testing.assert_allclose(gradient, 0., atol=1e-14)
    np.testing.assert_allclose(apply(jnp.ones(1)), [-2.4], atol=1e-13)


@pytest.mark.parametrize('basis,row,reference', [
    ('cc-pvdz', dict(symbols=['H', 'Br', 'N', 'H', 'H', 'H'],
        atomic_numbers=[1, 35, 7, 1, 1, 1], charge=0, spin=0, nelectron=46,
        coords_angstrom=[[.000015148, -.932477668, 0.], [-.000005844, .58174063, 0.],
            [.000026391, -2.578593783, 0.], [-.474353244, -2.931059587, .82164742],
            [-.474353244, -2.931059587, -.82164742], [.948782268, -2.931063413, 0.]]), -2629.1738131206034),
    ('6-31g(d,p)', dict(symbols=['C']*4+['H']*7, atomic_numbers=[6]*4+[1]*7,
        charge=0, spin=1, nelectron=31,
        coords_angstrom=[[.615889,.544785,.145402],[1.748652,-.348056,-.226713],
            [-.661464,-.199398,.410807],[-1.787248,-.038736,-.268849],
            [.436275,1.283902,-.650716],[.889557,1.129167,1.031913],
            [1.567569,-1.258807,-.783198],[2.773583,-.043031,-.060232],
            [-2.678019,-.609096,-.035092],[-1.858779,.672744,-1.085366],
            [-.625168,-.926451,1.218808]]), -155.48520611125923)])
def test_native_atomic_start_recovers_independently_checked_low_energy_branch(basis, row, reference):
    from nnao_reactivity.rescue import solve_native
    shells = traditional_shells(row, basis)
    data = make_inputs(row, shells)
    result = solve_native(data['overlap'], data['hcore'], data['factors'],
                          row['nelectron'], row['spin'], data['enuc'], row=row, atom_shells=shells)
    np.testing.assert_allclose(result['energy'], reference, atol=2e-8, rtol=0)
    assert result['orbital_residual'] < 1e-7
    assert result['stability']['stable']


def test_native_cp_rescue_keeps_ghost_basis_and_shared_auxiliary_space():
    from nnao_core.native import counterpoise
    from nnao_reactivity.rescue import solve_native
    row = dict(symbols=['He', 'He'], atomic_numbers=[2, 2],
               coords_angstrom=[[0., 0., 0.], [0., 0., 3.]])
    shells = [[[0, [.8, 1.]]], [[0, [.8, 1.]]]]
    result = counterpoise(row, shells, [[0], [1]], solver=solve_native)
    assert len(result['scf_diagnostics']) == 5
    assert all(s['stability']['stable'] for s in result['scf_diagnostics'])
    assert result['bsse_hartree'] > -1e-9
    np.testing.assert_allclose([s['nelectron'] for s in result['scf_diagnostics']], [4, 2, 2, 2, 2])


def test_ho2_negative_mode_escape_does_not_return_to_the_same_diis_saddle():
    from nnao_reactivity.rescue import solve_native
    row = dict(symbols=['O', 'O', 'H'], atomic_numbers=[8, 8, 1], charge=0,
        spin=1, nelectron=17, coords_angstrom=[[.055077, .712272, 0.],
                                            [.055077, -.603017, 0.], [-.881227, -.874042, 0.]])
    shells = traditional_shells(row, 'sto-6g')
    data = make_inputs(row, shells)
    result = solve_native(data['overlap'], data['hcore'], data['factors'], 17, 1, data['enuc'],
                          row=row, atom_shells=shells)
    # The original DIIS saddle is -149.5947884 Eh. An explicitly sampled
    # native determinant already has -149.6074060 Eh; a minimum must be lower.
    assert float(result['energy']) < -149.60740
    assert result['orbital_residual'] < 1e-7
    assert result['stability']['stable']
