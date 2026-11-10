"""RI Coulomb with independently supplied exact exchange; CPU float64."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf.scf.rks import RKSConfig, _make_jk_builder
from gradscf.scf.rsh import rsh_preset


def test_rks_facade_preserves_only_dfj_and_metric_threshold():
    mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g"), xc="pbe0")
    original = mf._scf_signature()
    assert mf.density_fit("weigend", only_dfj=True) is mf
    mf.df_tol = 1e-12
    cfg = mf._config()
    assert cfg.only_dfj and cfg.auxbasis == "weigend" and cfg.df_tol == 1e-12
    assert mf._scf_signature() != original
    assert mf.direct_scf() is mf and not mf._config().only_dfj


@pytest.mark.parametrize("xc", ["hf", "pbe0", "cam-b3lyp"])
def test_rks_ri_j_keeps_direct_exchange_and_physical_array_derivatives(monkeypatch, xc):
    from gradscf.integrals.backends.native import jk
    from gradscf.integrals.basis import prepare_basis
    from gradscf.integrals.plan import make_plan
    from gradscf.data.molecule import parse_molecule_spec

    top, params = prepare_basis(parse_molecule_spec("H 0 0 0; H 0 0 .8"), "sto-3g", cart=True)
    provider = jk.NativeDirectBasis(make_plan(top), params)
    raw = jnp.asarray([[[.7, .2], [.2, .4]], [[.1, -.3], [-.3, .5]]])
    exact = jnp.einsum("Qpq,Qrs->pqrs", raw, raw)
    calls = []

    def exact_action(self, density, *, with_j=True, with_k=True, omega=0., **controls):
        del self, controls
        calls.append((with_j, with_k, float(omega)))
        kernel = exact * (.3 if omega else 1.)
        j = jnp.einsum("pqrs,sr->pq", kernel, density) if with_j else jnp.zeros_like(density)
        k = jnp.einsum("prqs,rs->pq", kernel, density) if with_k else jnp.zeros_like(density)
        return j, k

    monkeypatch.setattr(jk.NativeDirectBasis, "get_jk", exact_action)
    cfg = RKSConfig(xc_spec=xc, jk_backend="df", auxbasis="weigend", only_dfj=True)
    preset = rsh_preset(xc)
    scale = 1. if preset is None else preset.default_sr_hf_fraction + .3*(
        preset.default_lr_hf_fraction-preset.default_sr_hf_fraction)

    def action(factors, density):
        builder = _make_jk_builder(None, cfg, df_factors=factors, direct_basis=provider)
        return jnp.stack(builder(density))

    def oracle(factors, density):
        ri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
        return jnp.stack([jnp.einsum("pqrs,sr->pq", ri, density),
                          scale*jnp.einsum("prqs,rs->pq", exact, density)])

    factors = raw*1.17
    density = jnp.asarray([[1.1, .2], [.2, .6]])
    np.testing.assert_allclose(action(factors, density), oracle(factors, density), atol=1e-12, rtol=1e-12)
    assert calls and all(not with_j and with_k for with_j, with_k, _ in calls)
    if preset is not None:
        assert {omega for _, _, omega in calls} == {0., preset.default_omega}
    direction = (.02*factors, -.03*density)
    actual = jax.jvp(action, (factors, density), direction)[1]
    expected = jax.jvp(oracle, (factors, density), direction)[1]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    objective = lambda fn, b, d: jnp.sum(fn(b, d)**2)
    actual = jax.grad(lambda b, d: objective(action, b, d), argnums=(0, 1))(factors, density)
    expected = jax.grad(lambda b, d: objective(oracle, b, d), argnums=(0, 1))(factors, density)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-10, rtol=1e-10)
