"""Matrix-free Super-CI on exact packed no-core operators."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm

from gradscf.mc import casci, csf, orbitals
from gradscf.mc.integrals import transform_active


@pytest.fixture(scope="module")
def large_packed_step():
    rng = np.random.default_rng(108)
    nmo, active = 35, (0, 2)
    pairs = orbitals._rotation_pairs(nmo, (), active)
    assert len(pairs) == 66

    rotation = np.linalg.qr(rng.normal(size=(nmo, nmo)))[0]
    metric_factor = np.linalg.cholesky(np.eye(nmo) + .1 * rotation)
    # Use a nonidentity metric and a non-contiguous active index set.
    overlap = metric_factor @ metric_factor.T
    c = np.linalg.solve(metric_factor.T, rotation)
    inverse = np.linalg.inv(c)

    gaps = np.linspace(.5, 2., nmo)
    gaps[list(active)] = [-1.5, -1.4]
    off_diagonal = rng.normal(scale=.015, size=(nmo, nmo))
    h = inverse.T @ (np.diag(gaps) + (off_diagonal + off_diagonal.T) * .5) @ inverse
    factors = rng.normal(scale=.013, size=(4, nmo, nmo))
    factors = (factors + factors.transpose(0, 2, 1)) * .5
    factors[0, 0, 2] = factors[0, 2, 0] = .35
    factors = np.einsum("pi,kpq,qj->kij", inverse, factors, inverse)
    p, q = np.tril_indices(nmo)
    packed_factors = factors[:, p, q]
    s, h, c, packed = map(jnp.asarray, (overlap, h, c, packed_factors.T @ packed_factors))

    options = {"eri_pair_matrix": packed, "max_memory": 8000.}
    operators = csf.make_csf_space(len(active), (1, 1), 0)
    hcas, gcas, constant = transform_active(h, c, core=(), active=active, **options)
    state = casci.solve_casci(hcas, gcas, operators, ecore=constant, with_response=False)
    assert bool(state.converged)
    return s, h, c, *operators.rdm12(state.ci_coeff), (), active, pairs, options


def test_large_packed_super_ci_uses_exact_blocks_with_davidson(large_packed_step, monkeypatch):
    s, h, c, d1, d2, core, active, pairs, options = large_packed_step
    # The AO-action route gives an independent reference at the same metric.
    with monkeypatch.context() as patch:
        patch.setattr(orbitals, "_packed_super_ci_blocks_fit", lambda *args: False)
        expected = orbitals._super_ci_step(*large_packed_step)

    def forbid_ao_action(*args):
        raise AssertionError("large packed steps should keep the MO-block action")

    monkeypatch.setattr(orbitals, "_super_ci_action", forbid_ao_action)
    actual = orbitals._super_ci_step(*large_packed_step)
    assert np.linalg.norm(expected) > .001
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-9)
    updated = c @ expm(orbitals._rotation_matrix(actual, pairs, c.shape[1], skew=True))
    np.testing.assert_allclose(updated.T @ s @ updated, np.eye(c.shape[1]), rtol=0., atol=2e-12)


def test_large_packed_super_ci_falls_back_under_a_tight_budget(large_packed_step, monkeypatch):
    s, h, c, d1, d2, core, active, pairs, options = large_packed_step
    # The asymmetric half transform makes vvaa fit at 16 MB. This tighter
    # cap still forces fallback for the balanced vaav block.
    tight = {**options, "max_memory": 8.}
    assert not orbitals._packed_super_ci_blocks_fit(h, c, d1, d2, len(active), tight)

    def forbid_blocks(*args):
        raise AssertionError("over-budget MO blocks must fall back to the AO action")

    monkeypatch.setattr(orbitals, "_no_core_super_ci_action", forbid_blocks)
    result = orbitals._super_ci_step(s, h, c, d1, d2, core, active, pairs, tight)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) > .001


def test_super_ci_reservation_accepts_packed_array_likes():
    h, c = jnp.eye(2), jnp.eye(2)
    d1, d2 = jnp.ones((1, 1)), jnp.ones((1, 1, 1, 1))
    packed = jnp.eye(3)
    expected = orbitals._packed_super_ci_resident_bytes(h, c, d1, d2, 1, packed)
    actual = orbitals._packed_super_ci_resident_bytes(h, c, d1, d2, 1, packed.tolist())
    assert actual == expected
