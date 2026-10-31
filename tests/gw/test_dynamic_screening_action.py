"""Exact dense and checked matrix-free products on both frequency axes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("retarded", [False, True])
def test_dynamic_screening_action_matches_dense_value_and_array_response(retarded, monkeypatch):
    from gradscf.gw import screened
    assert hasattr(screened, "screened_action"), "Dynamic GW needs a bounded screened action"
    from gradscf.solvers import LinearSolverConfig
    from gradscf.gw.polarizability import rho_response_iw, rho_response_real
    rng = np.random.default_rng(17)
    factors = jnp.asarray(rng.normal(size=(7, 2, 3))*.07)
    energy = jnp.array([-.9, -.6, .2, .7, 1.2])
    rhs = jnp.asarray(rng.normal(size=(7, 2))*.2)
    cfg = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13, restart=14)
    def reference(t):
        e, b = energy+t*jnp.arange(5)*.01, factors*(1+.1*t)
        pi = (rho_response_real(.3, e, b, eta=.02+.001*t) if retarded else
              rho_response_iw(.3, e, b, spin_factor=2.))
        return jnp.linalg.solve(jnp.eye(7)-pi, pi@(rhs+t*.02))
    def actual(t):
        e, b = energy+t*jnp.arange(5)*.01, factors*(1+.1*t)
        return screened.screened_action(.3, ((e, b, 2.),), rhs+t*.02,
            eta=.02+.001*t if retarded else None, config=cfg)
    expected, derivative = jax.jvp(reference, (0.,), (1.,))
    # Matrix-free mode must not call an auxiliary-matrix constructor.
    monkeypatch.setattr(screened, "rho_response_real", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense Pi")), raising=False)
    value, tangent = jax.jit(lambda t: jax.jvp(actual, (t,), (jnp.ones_like(t),)))(0.)
    np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(tangent, derivative, atol=1e-10, rtol=1e-8)
    gradient = jax.jit(jax.grad(lambda t: jnp.real(actual(t)).sum()))(0.)
    np.testing.assert_allclose(gradient, derivative.real.sum(), atol=1e-10, rtol=1e-8)
    curvature = jax.jit(jax.grad(jax.grad(lambda t: jnp.real(actual(t)).sum())))(0.)
    expected_curvature = jax.grad(jax.grad(lambda t: jnp.real(reference(t)).sum()))(0.)
    np.testing.assert_allclose(curvature, expected_curvature, atol=1e-9, rtol=1e-7)


def test_explicit_direct_configuration_is_rejected_instead_of_ignored():
    from gradscf.gw.screened import screened_action
    from gradscf.solvers import LinearSolverConfig
    with pytest.raises(ValueError, match="GMRES"):
        screened_action(.3, ((jnp.array([-.5, .5]), jnp.ones((2, 1, 1))*.1, 2.),),
                        jnp.ones(2), config=LinearSolverConfig(method="direct", max_dense=1))
