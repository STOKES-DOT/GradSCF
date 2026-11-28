"""Exact reuse of fixed-Hamiltonian intermediates during CASCI."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.mc.casci import solve_casci
from gradscf.mc.csf import CSFOperators, make_csf_space
from gradscf.solvers import EigenSolverConfig

from .test_csf import model


@pytest.mark.parametrize("pair_block_size", [None, 3])
def test_prepared_sigma_reuses_projected_integrals_with_live_response(pair_block_size):
    space = make_csf_space(4, (2, 1), 1)
    h, eri = map(jnp.asarray, model(4))
    vectors = jnp.sin(.3 + jnp.arange(2 * space.size)).reshape(2, space.size)
    h += .01 * jnp.sin(jnp.arange(h.size).reshape(h.shape))
    eri += .01 * jnp.cos(jnp.arange(eri.size).reshape(eri.shape))

    @jax.jit
    def prepared(h, eri, vectors):
        action = space._prepare_symmetric_action(h, eri, pair_block_size=pair_block_size)
        return jax.vmap(action)(vectors)

    def ordered(h, eri, vectors):
        h = .5 * (h + h.T)
        eri = .5 * (eri + eri.transpose(1, 0, 2, 3))
        eri = .5 * (eri + eri.transpose(0, 1, 3, 2))
        eri = .5 * (eri + eri.transpose(2, 3, 0, 1))
        return jax.vmap(lambda c: space.matvec(h, eri, c))(vectors)

    primals = (h, eri, vectors)
    tangents = tuple(jnp.cos(x) for x in primals)
    actual = jax.jvp(prepared, primals, tangents)
    expected = jax.jvp(ordered, primals, tangents)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)
    probe = jnp.cos(vectors)
    pullback = jax.vjp(prepared, *primals)[1](probe)
    reference = jax.vjp(ordered, *primals)[1](probe)
    for result, target in zip(pullback, reference):
        np.testing.assert_allclose(result, target, atol=2e-12, rtol=0.)


def test_davidson_prepares_the_action_once_before_repeated_applications(monkeypatch):
    space = make_csf_space(3, (1, 1), 0)
    h, eri = map(jnp.asarray, model(3))
    config = EigenSolverConfig(method="davidson", atol=1e-10)
    prepare = CSFOperators._prepare_symmetric_action
    preparations, applications = [], []

    def record(*args, **kwargs):
        preparations.append(None)
        action = prepare(*args, **kwargs)

        def apply(c):
            applications.append(None)
            return action(c)

        return apply

    monkeypatch.setattr(CSFOperators, "_prepare_symmetric_action", record)
    with jax.disable_jit():
        result = solve_casci(h, eri, space, config=config, with_response=False)
    assert bool(result.converged)
    assert len(preparations) == 1
    assert len(applications) > 1
    ordered = space.matmat(h, eri, jnp.eye(space.size))
    np.testing.assert_allclose(result.active_energy, np.linalg.eigvalsh(ordered)[0],
                               atol=1e-10, rtol=0.)
    np.testing.assert_allclose(result.ci_residual,
                               jnp.max(jnp.abs(ordered @ result.ci_coeff
                                               - result.active_energy * result.ci_coeff)),
                               atol=2e-12, rtol=0.)
