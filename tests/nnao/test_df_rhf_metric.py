"""An invertible AO change must preserve the implicit RHF energy and response."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.training.objective import _df_rhf_solver
from gradscf.integrals.molecular.jk import build_jk_from_df


def _problem():
    rng = np.random.default_rng(24)
    left = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    right = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    transform = jnp.asarray(left @ np.diag([1., 1., 1., 6e-4]) @ right.T)
    symmetric = lambda x: (x + np.swapaxes(x, -1, -2)) / 2
    ds = jnp.asarray(symmetric(rng.normal(size=(4, 4))) * .03)
    dh = jnp.asarray(symmetric(rng.normal(size=(4, 4))) * .1)
    factors = jnp.asarray(symmetric(rng.normal(size=(3, 4, 4))) * .04)

    def physical(theta, changed):
        s = jnp.eye(4) + theta * ds
        h = jnp.diag(jnp.array([-1.2, -.7, .4, .9])) + theta * dh
        b = factors * jnp.exp(.1 * theta)
        return tuple(transform.T @ x @ transform for x in (s, h, b)) if changed else (s, h, b)

    return physical


def test_ill_conditioned_invertible_basis_preserves_scf_and_implicit_response():
    physical = _problem()
    value, _ = _df_rhf_solver.__wrapped__(4, 1e-8, .2)
    theta = jnp.asarray(.07)
    losses = []
    for changed in (False, True):
        loss = lambda t: value(physical(t, changed), .3)[0]
        energy, info = value(physical(theta, changed), .3)
        assert bool(info['converged']) and bool(info['stable'])
        assert float(info['fixed_point_residual']) < 1e-7
        assert float(info['min_overlap_eigenvalue']) > 1e-8
        np.testing.assert_allclose(info['min_overlap_eigenvalue'],
            jnp.linalg.eigvalsh(physical(theta, changed)[0])[0], atol=1e-12, rtol=0)
        derivative = jax.grad(loss)(theta)
        curvature = jax.jvp(jax.grad(loss), (theta,), (jnp.ones_like(theta),))[1]
        losses.append(np.asarray([energy, derivative, curvature]))
    assert np.isfinite(losses).all()
    np.testing.assert_allclose(losses[1], losses[0], atol=2e-7, rtol=2e-6)

    # Independently unroll the weakly interacting reference, using symmetric
    # orthogonalization. This checks Pulay derivatives as well as invariance.
    def unrolled(t):
        s, h, b = physical(t, False)
        eigenvalues, vectors = jnp.linalg.eigh(s)
        x = (vectors * eigenvalues**-.5) @ vectors.T

        def step(_, density):
            j, k = build_jk_from_df(b, density)
            _, orbitals = jnp.linalg.eigh(x.T @ (h+j-.5*k) @ x)
            occupied = (x @ orbitals)[:, :2]
            return 2 * occupied @ occupied.T

        density = jax.lax.fori_loop(0, 16, step, jnp.zeros_like(h))
        j, k = build_jk_from_df(b, density)
        return jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+.3

    reference = np.asarray([unrolled(theta), jax.grad(unrolled)(theta),
        jax.jvp(jax.grad(unrolled), (theta,), (jnp.ones_like(theta),))[1]])
    np.testing.assert_allclose(losses[0], reference, atol=2e-8, rtol=2e-6)
