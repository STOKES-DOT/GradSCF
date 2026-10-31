"""Spin execution requests only the physical operators its objective needs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import gks, uks
from gradscf.integrals.molecular import jk


def inputs():
    rng = np.random.default_rng(1031)
    factors = rng.normal(size=(4, 3, 3))*.2
    factors = (factors+factors.transpose(0, 2, 1))*.5
    eri = jnp.asarray(np.einsum('Qpq,Qrs->pqrs', factors, factors))
    p, q = np.tril_indices(3)
    packed = eri[p, q][:, p, q]
    spinor = rng.normal(size=(6, 2))+1j*rng.normal(size=(6, 2))
    spinor, _ = np.linalg.qr(spinor)
    density = jnp.asarray(spinor@spinor.conj().T)
    ao = jnp.asarray(rng.normal(size=(7, 3))*.4)
    return eri, packed, density, dict(hcore=-jnp.eye(3), ao=ao,
        ao_deriv1=jnp.stack([ao, .1*ao, -.07*ao, .03*ao]),
        grid_weights=jnp.arange(1., 8.)*.02, nuclear_repulsion=jnp.asarray(.3))


@pytest.mark.parametrize('xc,collinear', [('pbe', 'col'), ('pbe0', 'col'),
                                          ('lda_x', 'ncol'), ('hf', 'col')])
def test_gks_selects_j_and_spinor_k_independently(monkeypatch, xc, collinear):
    eri, packed, density, physical = inputs()
    config = gks.GKSConfig(xc_spec=xc, collinear=collinear)
    expected = gks.generalized_energy_and_fock(density=density, eri=eri,
                                             config=config, **physical)
    calls, original = [], jk.build_jk_from_packed
    def selected(integrals, dm, **options):
        calls.append((dm.shape, options.get('with_j', True), options.get('with_k', True)))
        return original(integrals, dm, **options)
    monkeypatch.setattr(jk, 'build_jk_from_packed', selected)
    actual = gks.generalized_energy_and_fock(density=density, eri=packed,
                                           config=config, **physical)
    expected_calls = [((3, 3), True, False)]
    if xc in ('hf', 'pbe0'):
        expected_calls.append(((2, 2, 3, 3), False, True))
    assert calls == expected_calls
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('xc,collinear', [('pbe0', 'col'), ('lda_x', 'ncol')])
def test_gks_energy_only_does_not_build_xc_potentials(monkeypatch, xc, collinear):
    from gradscf.scf.orbital_optimization import _problem_functions
    eri, _, density, physical = inputs()
    config = gks.GKSConfig(xc_spec=xc, collinear=collinear)
    expected = gks.generalized_energy_and_fock(density=density, eri=eri,
                                             config=config, **physical)[0]
    def forbidden(*args, **kwargs):
        raise AssertionError('Energy-only execution must not request XC potentials')
    monkeypatch.setattr(gks, '_unrestricted_vxc_matrices', forbidden)
    monkeypatch.setattr(gks, 'grid_value_and_grad', forbidden)
    actual = gks.generalized_energy_and_fock(density=density, eri=eri,
        config=config, compute_potential=False, **physical)
    np.testing.assert_allclose(actual[0], expected, atol=2e-12, rtol=2e-12)
    assert actual[2] is None
    functions = _problem_functions('gks', xc, collinear, ((1., 1., 0., 0., 0., 0.),))
    dimension, _, density_from, evaluate, energy = functions
    args = dict(physical, eri=eri, grid_ao=None)
    base = jnp.eye(6, dtype=jnp.complex128)[None]
    scalar = energy(jnp.zeros(dimension), base, args)
    direct = evaluate(density_from(base), args, compute_potential=False)[0]
    np.testing.assert_allclose(scalar, direct, atol=2e-12)


@pytest.mark.parametrize('layout', ['full', 's4', 's8'])
def test_uks_exchange_uses_one_spin_batch(monkeypatch, layout):
    eri, packed, density, _ = inputs()
    integrals = {'full': eri, 's4': packed,
                 's8': packed[np.tril_indices(packed.shape[0])]}[layout]
    a, b = density[:3, :3].real, density[3:, 3:].real
    expected = (a+b, uks._build_jk(eri, a+b, with_k=False)[0],
                uks._build_jk(eri, a, with_j=False)[1],
                uks._build_jk(eri, b, with_j=False)[1])
    calls, original = [], uks._build_jk
    def selected(integrals, dm, **options):
        calls.append((dm.shape, options.get('with_j', True), options.get('with_k', True)))
        return original(integrals, dm, **options)
    monkeypatch.setattr(uks, '_build_jk', selected)
    actual = uks._spin_jk_matrices(integrals, a, b)
    assert calls == [((3, 3), True, False), ((2, 3, 3), False, True)]
    for got, reference in zip(actual, expected):
        np.testing.assert_allclose(got, reference, atol=2e-12, rtol=2e-12)
