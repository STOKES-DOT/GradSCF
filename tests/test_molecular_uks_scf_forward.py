"""UKS numerical seeds use the existing DIIS driver and keep spin topology."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy


@pytest.mark.parametrize('empty_beta', [False, True])
def test_uks_scf_seed_relays_independent_occupations_and_failure(monkeypatch, empty_beta):
    from gradscf.scf import uks
    # Routing fixture only; physical convergence/derivatives are checked below.
    mf = dft.UKS(gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', cart=True,
                       charge=1 if empty_beta else -1, spin=1), xc='pbe0')
    mf.converged = True
    mf.mo_coeff = np.stack([np.eye(2), np.array([[0., 1.], [1., 0.]])])
    mf.mo_occ = np.array([[0., 1.], [0., 0.]]) if empty_beta else np.array([[1., 1.], [0., 1.]])
    model = MolecularEnergy(mf, forward='scf', jk_backend='direct')
    fields = dict(overlap=jnp.eye(2), hcore=jnp.eye(2), eri=None,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
        grid_weights=jnp.zeros(0), nuclear_repulsion=jnp.asarray(0.))
    success = [True]
    def producer(**kwargs):
        assert (kwargs['nalpha'], kwargs['nbeta']) == tuple(int(x.sum()) for x in mf.mo_occ)
        assert kwargs['config'].xc_spec == 'pbe0'
        ordered = [c[:, np.r_[np.flatnonzero(o), np.flatnonzero(o == 0)]]
                   for c, o in zip(mf.mo_coeff, mf.mo_occ)]
        np.testing.assert_array_equal(kwargs['init_mo_coeff_alpha'], ordered[0])
        np.testing.assert_array_equal(kwargs['init_mo_coeff_beta'], ordered[1])
        return SimpleNamespace(converged=jnp.asarray(success[0]),
            mo_coeff_alpha=jnp.asarray(ordered[0]), mo_coeff_beta=jnp.asarray(ordered[1]))
    monkeypatch.setattr(uks, 'run_uks_from_integrals', producer)
    np.testing.assert_array_equal(model._scf_seed(fields), mf.mo_coeff)
    success[0] = False
    assert not np.isfinite(np.asarray(model._scf_seed(fields))).any()


@pytest.mark.parametrize('xc,empty_beta', [('svwn', False), ('pbe', False),
                                         ('pbe0', True), ('wb97xd', True)])
def test_uks_scf_forward_keeps_relaxed_geometry_and_field_response(xc, empty_beta):
    atom = 'H 0 0 -.7; H 0 0 .7' if empty_beta else 'O 0 0 0; H 0 0 1.83'
    mf = dft.UKS(gto.M(atom=atom, basis='sto-3g', unit='Bohr', cart=True,
        charge=1 if empty_beta else 0, spin=1), xc=xc, grids_level=0,
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=200).run()
    assert mf.converged
    direct = MolecularEnergy(mf, forward='scf', jk_backend='direct')
    original = MolecularEnergy(mf, jk_backend='direct')
    r, field = direct.coordinates, jnp.array([0., 0., .001])
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    objective = lambda q, model: model(r+q*direction, field)
    value_grad = jax.jit(jax.value_and_grad(lambda q: objective(q, direct)))
    actual = value_grad(jnp.asarray(0.))
    expected = jax.jit(jax.value_and_grad(lambda q: objective(q, original)))(jnp.asarray(0.))
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=2e-7)
    hvp = jax.jit(jax.grad(jax.grad(lambda q: objective(q, direct))))(jnp.asarray(0.))
    for step in (3e-4, 1e-4):
        plus, minus = value_grad(jnp.asarray(step)), value_grad(jnp.asarray(-step))
        np.testing.assert_allclose(actual[1], (plus[0]-minus[0])/(2*step), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(hvp, (plus[1]-minus[1])/(2*step), atol=4e-6, rtol=2e-5)
