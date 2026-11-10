"""Synthetic basis-parameter training through fully relaxed implicit HF.

H2 at 1.6 Bohr, float64, three primitives contracted into one AO per atom.
Only the first atom's raw coefficients and log exponents are trained, so
radial changes cannot be absorbed into a complete primitive AO span. This
is not a MACE/model training test or a one-electron mixed-force AD test.
S/T/V, nuclear repulsion and the moving AO metric enter every energy solve;
nuclei remain fixed while the basis changes. Energies are in Hartree.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.linalg import eigh

from gradscf import integrals, scf
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan


@pytest.fixture(scope="module")
def problem():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 1.6",
        {"H": [[0, [2.0, .4], [.6, .7], [.15, .2]]]},
        unit="Bohr", cart=True)
    plan = integrals.make_plan(topology)
    overlap = plan.evaluate("overlap", parameters)
    hcore = plan.evaluate("kinetic", parameters) + plan.evaluate("nuclear", parameters)
    _, seed = eigh(np.asarray(hcore), np.asarray(overlap))
    occupations = np.array([[1., 0.], [1., 0.]])
    point = jnp.concatenate((parameters.coefficients[0].ravel(), jnp.zeros(3)))

    def bind(value):
        return replace(parameters,
            coefficients=(value[:3, None], *parameters.coefficients[1:]),
            exponents=(parameters.exponents[0]*jnp.exp(value[3:]),
                       *parameters.exponents[1:]))

    def energy(value, *, direct):
        current = bind(value)
        s = plan.evaluate("overlap", current)
        h = plan.evaluate("kinetic", current) + plan.evaluate("nuclear", current)
        state = scf.minimize_roks_from_integrals(
            overlap=s, hcore=h,
            eri=None if direct else plan.evaluate("eri", current),
            direct_basis=NativeDirectBasis(plan, current) if direct else None,
            nuclear_repulsion=scf.nuclear_repulsion_energy(
                current.nuclear_coords, jnp.asarray(topology.nuclear_charges)),
            ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
            grid_weights=jnp.zeros(0), mo_coeff=seed, mo_occ=occupations,
            xc_spec="hf", gradient_mode="implicit", orthonormalize_initial=True,
            max_iterations=100, gradient_tolerance=1e-10,
            direct_scf_tol=0., direct_scf_workers=2, max_memory=64.)
        electrons = jnp.einsum("spq,qp->s", state.density_matrix, s)
        return state.total_energy, (state.stationary, state.gradient_norm, electrons)

    return plan, bind, point, energy


def checked(value, info):
    stationary, norm, electrons = info
    assert bool(stationary) and float(norm) <= 1e-10
    assert np.isfinite(float(value))
    np.testing.assert_allclose(electrons, np.ones(2), atol=2e-10, rtol=0.)
    return value


def forbid_direct_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def evaluate(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Direct basis training must not assemble AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", evaluate)


def test_relaxed_basis_energy_gradient_matches_full_and_reconverged_fd(problem, monkeypatch):
    _, _, point, energy = problem
    gradients, values, differences = [], [], []
    for direct in (False, True):
        if direct:
            forbid_direct_eri(monkeypatch)
        action = lambda x: energy(x, direct=direct)
        (value, info), gradient = jax.jit(jax.value_and_grad(action, has_aux=True))(point)
        values.append(checked(value, info))
        assert np.isfinite(np.asarray(gradient)).all()
        assert np.linalg.norm(np.asarray(gradient[:3])) > 1e-5
        assert np.linalg.norm(np.asarray(gradient[3:])) > 1e-5
        gradients.append(gradient)
        scalar = jax.jit(action)
        for step in (2e-4, 1e-4):
            # Every endpoint reconstructs its inputs and reconverges orbitals.
            fd = jnp.stack([(checked(*scalar(point+step*d))
                             - checked(*scalar(point-step*d)))/(2*step)
                            for d in jnp.eye(point.size)])
            differences.append(fd)
            np.testing.assert_allclose(gradient, fd, atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(values[1], values[0], atol=2e-10, rtol=0.)
    np.testing.assert_allclose(gradients[1], gradients[0], atol=3e-8, rtol=3e-7)
    for fd in differences[1:]:
        np.testing.assert_allclose(fd, differences[0], atol=3e-7, rtol=3e-6)


def test_synthetic_energy_loss_updates_live_coefficients_and_exponents(problem, monkeypatch):
    plan, bind, point, energy = problem
    reference, info = jax.jit(lambda x: energy(x, direct=False))(point)
    checked(reference, info)
    target = float(reference) - .05  # Synthetic target, not an external HF reference.
    forbid_direct_eri(monkeypatch)

    def loss(value):
        total, info = energy(value, direct=True)
        return (total-target)**2, (total, info)

    evaluate = jax.jit(jax.value_and_grad(loss, has_aux=True))
    (initial_loss, (initial, info)), gradient = evaluate(point)
    checked(initial, info)
    assert np.isfinite(np.asarray(gradient)).all() and np.isfinite(float(initial_loss))
    optimizer = optax.sgd(.01)
    updates, _ = optimizer.update(gradient, optimizer.init(point), point)
    updated = optax.apply_updates(point, updates)
    (updated_loss, (total, info)), new_gradient = evaluate(updated)
    checked(total, info)
    assert np.isfinite(np.asarray(new_gradient)).all() and float(updated_loss) < float(initial_loss)
    assert np.linalg.norm(np.asarray(updated[:3]-point[:3])) > 1e-9
    assert np.linalg.norm(np.asarray(updated[3:]-point[3:])) > 1e-9
    assert np.all(np.asarray(bind(updated).exponents[0]) > 0.)
    old_s = plan.evaluate("overlap", bind(point))
    new_s = plan.evaluate("overlap", bind(updated))
    assert np.linalg.norm(np.asarray(new_s-old_s)) > 1e-9
