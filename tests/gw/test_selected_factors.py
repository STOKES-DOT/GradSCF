"""Early G/W/QP windows preserve raw independent factor products and AD."""
from importlib import import_module, util
import jax
import jax.numpy as jnp
import numpy as np


def test_selected_factors_equal_full_then_slice_and_preserve_array_ad():
    assert util.find_spec("gradscf.gw.integrals") is not None, "GW needs selected factor preparation"
    prepare = import_module("gradscf.gw.integrals").prepare_selected_factors
    from gradscf.gw.g0w0 import _mo_factors, _exchange_mo
    rng = np.random.default_rng(13)
    factors = jnp.asarray(rng.normal(size=(5, 5, 5))*.1)  # intentionally nonsymmetric
    coeff = jnp.asarray(np.linalg.qr(rng.normal(size=(5, 5)))[0])
    target, green, occupied, virtual = (4, 1), (3, 0, 2), (1, 0), (4, 2)
    def run(t, reference=False):
        b, c = factors*(1+.1*t), coeff+t*.01
        if reference:
            full = _mo_factors(b, c)
            return dict(b_ov=full[:, jnp.array(occupied)[:, None], jnp.array(virtual)[None, :]],
                b_pm=full[:, jnp.array(target)[:, None], jnp.array(green)[None, :]],
                b_mp=full[:, jnp.array(green)[:, None], jnp.array(target)[None, :]].transpose(0, 2, 1),
                exchange=jnp.diag(_exchange_mo(full, 2))[jnp.array(target)])
        return prepare(b, c, nocc=2, orbs=target, g_orbitals=green,
                       screening_occupied=occupied, screening_virtual=virtual,
                       max_memory=10.)
    actual, tangent = jax.jit(lambda t: jax.jvp(run, (t,), (jnp.ones_like(t),)))(0.)
    expected, derivative = jax.jvp(lambda t: run(t, True), (0.,), (1.,))
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(tangent[name], derivative[name], atol=1e-11, rtol=1e-9)
