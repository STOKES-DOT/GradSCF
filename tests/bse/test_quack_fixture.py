"""Offline regression against an actually executed pinned QuAcK Fortran oracle."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("singlet", [True, False])
def test_static_tda_matches_quack_fixture(singlet):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening

    path = Path(__file__).parent / "data" / "quack_static_seed83.npz"
    with np.load(path) as data:
        qp, e, factors = [
            jnp.asarray(data[name])
            for name in ("qp_energy", "screening_energy", "factors")
        ]
        expected = data["a_singlet" if singlet else "a_triplet"]
    space = bse.make_bse_space(5, 2)
    state = build_static_screening(
        e, factors, occupied=space.occupied, virtual=space.virtual
    )
    operator = bse.build_tda_operator(
        qp, factors, space, state, singlet=singlet, block_size=2
    )
    actual = np.asarray(operator.apply(jnp.eye(space.size)))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
    np.testing.assert_allclose(operator.diagonal, np.diag(expected), atol=2e-13, rtol=0)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(actual), np.linalg.eigvalsh(expected), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("singlet", [True, False])
@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_full_bse_matches_executed_quack_coupling_fixture(singlet, method):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening

    path = Path(__file__).parent / "data/quack_full_static_seed83.npz"
    label = "singlet" if singlet else "triplet"
    with np.load(path) as data:
        qp, e, l = [
            jnp.asarray(data[name])
            for name in ("qp_energy", "screening_energy", "factors")
        ]
        expected_b = data[f"b_{label}"]
        expected_roots = data[f"full_roots_{label}"]
    space = bse.make_bse_space(5, 2)
    state = build_static_screening(e, l, occupied=space.occupied, virtual=space.virtual)
    _, bo = bse.build_bse_operators(qp, l, space, state, singlet=singlet, block_size=2)
    np.testing.assert_allclose(bo.apply(jnp.eye(6)), expected_b, atol=2e-13, rtol=0)
    out = bse.run_bse(
        qp,
        e,
        l,
        space,
        config=bse.BSEConfig(tda=False, solver=method, nroots=6, singlet=singlet),
    )
    np.testing.assert_allclose(
        out.excitation_energies, expected_roots, atol=2e-12, rtol=0
    )
