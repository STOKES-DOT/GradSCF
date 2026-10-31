"""Pure classic DFT packed fallbacks request only the Coulomb operator.

Permutation-symmetric algebraic ERIs isolate SCF dispatch from XC/grid work.
The original nonsymmetric trial densities and their AD directions stay live.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.xc import hybrid_coeff
from gradscf.scf import rks, uks


@pytest.fixture(params=["s4", "s8"])
def packed_problem(request):
    rng = np.random.default_rng(937)
    nao = 3
    pairs = nao*(nao+1)//2
    factors = rng.normal(size=(pairs, pairs))
    pair = factors.T @ factors / pairs
    indices = np.arange(nao)
    hi = np.maximum(indices[:, None], indices)
    lo = np.minimum(indices[:, None], indices)
    pair_ids = hi*(hi+1)//2 + lo
    full = pair[pair_ids[:, :, None, None], pair_ids[None, None, :, :]]
    packed = pair if request.param == "s4" else pair[np.tril_indices(pairs)]
    density = rng.normal(size=(2, nao, nao))
    return jnp.asarray(packed), jnp.asarray(full), jnp.asarray(density)


def _record_packed_selection(monkeypatch):
    from gradscf.integrals.molecular import jk

    original = jk.build_jk_from_packed
    calls = []

    def record(eri, density, **kwargs):
        calls.append((kwargs.get("with_j", True), kwargs.get("with_k", True)))
        return original(eri, density, **kwargs)

    monkeypatch.setattr(jk, "build_jk_from_packed", record)
    return calls


def _check_density_response(action, oracle, density):
    observed = jax.jit(action)(density)
    expected = oracle(density)
    for actual, reference in zip(observed, expected):
        np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-12)
    direction = jnp.cos(density)*.07
    tangent = jax.jit(lambda d: jax.jvp(action, (d,), (direction,))[1])(density)
    reference_tangent = jax.jvp(oracle, (density,), (direction,))[1]
    for actual, reference in zip(tangent, reference_tangent):
        np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-12)
    cotangent = tuple(jnp.sin(jnp.arange(a.size).reshape(a.shape)+.3) for a in expected)
    gradient = jax.jit(lambda d: jax.vjp(action, d)[1](cotangent)[0])(density)
    reference_gradient = jax.vjp(oracle, density)[1](cotangent)[0]
    np.testing.assert_allclose(gradient, reference_gradient, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("xc", ["svwn", "pbe"])
def test_rks_pure_packed_eri_fallback_requests_only_coulomb(packed_problem, monkeypatch, xc):
    packed, full, densities = packed_problem
    assert hybrid_coeff(xc) == 0.
    calls = _record_packed_selection(monkeypatch)
    action = rks._make_jk_builder(packed, rks.RKSConfig(xc_spec=xc), with_k=False)

    def oracle(density):
        return (jnp.einsum("pqrs,rs->pq", full, density, precision="highest"),
                jnp.zeros_like(density))

    _check_density_response(action, oracle, densities[0])
    assert calls and all(flags == (True, False) for flags in calls), calls


@pytest.mark.parametrize("xc", ["svwn", "pbe"])
def test_uks_pure_packed_total_density_requests_only_coulomb(packed_problem, monkeypatch, xc):
    packed, full, densities = packed_problem
    with_k = bool(abs(float(hybrid_coeff(xc))) > 1e-14)
    assert not with_k
    calls = _record_packed_selection(monkeypatch)
    action = lambda density: uks._spin_jk_matrices(
        packed, density[0], density[1], with_k=with_k)

    def oracle(density):
        total = density[0]+density[1]
        coulomb = jnp.einsum("pqrs,rs->pq", full, total, precision="highest")
        zero = jnp.zeros_like(total)
        return total, coulomb, zero, zero

    _check_density_response(action, oracle, densities)
    assert calls and all(flags == (True, False) for flags in calls), calls
