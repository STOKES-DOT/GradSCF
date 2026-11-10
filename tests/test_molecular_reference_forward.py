"""Split SCF/implicit-response execution keeps full orbital and metric response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy


@pytest.mark.parametrize("xc", ["pbe0", "wb97xd"])
def test_reference_forward_has_full_response_and_dynamic_solution(monkeypatch, xc):
    from gradscf.scf import rks
    mf = dft.RKS(gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True),
        xc=xc, grids_level=0, conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=200).run()
    local = MolecularEnergy(mf, forward="reference", jk_backend="direct", gradient_tolerance=1e-9)
    full = MolecularEnergy(mf, forward="scf", jk_backend="direct", gradient_tolerance=1e-9)
    coordinates = full.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    field = jnp.array([0., 0., .001])
    solve = jax.jit(lambda f: full.solve(coordinates, f))
    first, second = solve(field), solve(2*field)
    assert bool(first.stationary) and bool(second.stationary)
    scalar = lambda q: full(coordinates+q*direction, field)
    complete = jax.jit(jax.value_and_grad(scalar))
    expected = complete(jnp.asarray(0.))
    expected_hvp = jax.jit(jax.grad(jax.grad(scalar)))(jnp.asarray(0.))
    endpoints = [(complete(jnp.asarray(step)), complete(jnp.asarray(-step))) for step in (3e-4, 1e-4)]

    def forbidden(*args, **kwargs):
        raise AssertionError("A supplied converged solution must not run SCF again")
    monkeypatch.setattr(rks, "run_rks_from_integrals_traceable", forbidden)
    value = lambda q, c, f: local(coordinates+q*direction, f, solution=c)
    calculate = jax.jit(jax.value_and_grad(value, argnums=0))
    actual = calculate(jnp.asarray(0.), first.mo_coeff, field)
    hvp = jax.jit(lambda q, c: jax.jvp(jax.grad(lambda x: value(x, c, field)),
        (q,), (jnp.ones_like(q),))[1])(jnp.asarray(0.), first.mo_coeff)
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(hvp, expected_hvp, atol=3e-7, rtol=2e-6)
    for step, (plus, minus) in zip((3e-4, 1e-4), endpoints):
        np.testing.assert_allclose(actual[1], (plus[0]-minus[0])/(2*step), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(hvp, (plus[1]-minus[1])/(2*step), atol=4e-6, rtol=2e-5)
    entries = calculate._cache_size()
    updated = calculate(jnp.asarray(0.), second.mo_coeff, 2*field)
    assert calculate._cache_size() == entries
    np.testing.assert_allclose(updated[0], second.total_energy, atol=3e-10, rtol=0)
    assert abs(float(updated[0]-actual[0])) > 1e-8
    # Reusing the old numerical solution away from its stationary point is
    # rejected; finite differences above always rerun the complete SCF.
    stale = jax.jit(lambda q: value(q, first.mo_coeff, field))(jnp.asarray(.01))
    assert not np.isfinite(float(stale))


def test_reference_mode_requires_implicit_and_solution_is_explicit():
    from gradscf import scf
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True)).run()
    with pytest.raises(ValueError, match="implicit"):
        MolecularEnergy(mf, forward="reference", mode="explicit")
    model = MolecularEnergy(mf, forward="scf")
    with pytest.raises(ValueError, match="reference"):
        model(model.coordinates, solution=mf.mo_coeff)


def test_reference_mode_keeps_unrestricted_empty_spin_response():
    from gradscf import scf
    mf = scf.UHF(gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g",
                       unit="Bohr", charge=1, spin=1, cart=True),
                 conv_tol=1e-12, conv_tol_grad=1e-10).run()
    full = MolecularEnergy(mf, jk_backend="direct")
    local = MolecularEnergy(mf, forward="reference", jk_backend="direct")
    r, field = full.coordinates, jnp.array([0., 0., .001])
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    state = jax.jit(lambda x: full.solve(x, field))(r)
    assert bool(state.stationary)
    complete = jax.jit(jax.value_and_grad(lambda q: full(r+q*direction, field)))
    expected = complete(jnp.asarray(0.))
    calculate = jax.jit(jax.value_and_grad(
        lambda q, c: local(r+q*direction, field, solution=c), argnums=0))
    actual = calculate(jnp.asarray(0.), state.mo_coeff)
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=2e-7)
    for step in (3e-4, 1e-4):
        finite = (complete(jnp.asarray(step))[0]-complete(jnp.asarray(-step))[0])/(2*step)
        np.testing.assert_allclose(actual[1], finite, atol=3e-6, rtol=2e-5)
