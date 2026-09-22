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
