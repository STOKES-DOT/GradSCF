"""Full BSE Davidson uses common metric response without materializing A/B."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_full import model


@pytest.mark.parametrize("singlet", [True, False])
def test_matrix_free_full_bse_matches_dense(singlet):
    from gradscf import bse

    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    dense = bse.run_bse(
        qp,
        e,
        l,
        space,
        config=bse.BSEConfig(nroots=3, tda=False, solver="dense", singlet=singlet),
    )
    cfg = bse.BSEConfig(
        nroots=3, tda=False, solver="davidson", singlet=singlet, max_dense=1
    )
    out = jax.jit(lambda q: bse.run_bse(q, e, l, space, config=cfg))(qp)
    np.testing.assert_allclose(
        out.excitation_energies, dense.excitation_energies, atol=2e-9, rtol=0
    )
    np.testing.assert_allclose(
        bse.oscillator_strengths(out, d, space),
        bse.oscillator_strengths(dense, d, space),
        atol=2e-8,
        rtol=0,
    )
    assert not out.stability_certified
    assert dense.stability_certified
    assert np.all(out.response_valid)


def test_bse_davidson_first_order_optical_response():
    from gradscf import bse

    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=2, tda=False, solver="davidson", max_dense=1)

    def f(t):
        out = bse.run_bse(
            qp * (1 + 0.03 * t),
            e * (1 - 0.05 * t),
            l * (1 + 0.2 * t),
            space,
            config=cfg,
        )
        return jnp.sum(
            out.excitation_energies * jnp.array([0.3, 0.8])
            + bse.oscillator_strengths(out, d * (1 + 0.1 * t), space)
        )

    grad = jax.jit(jax.grad(f))(0.0)
    np.testing.assert_allclose(grad, (f(1e-4) - f(-1e-4)) / 2e-4, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(jax.jvp(f, (0.0,), (1.0,))[1], grad, atol=1e-9)
