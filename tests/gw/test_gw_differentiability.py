"""Differentiability acceptance tests for the GW module.

Verifies, on H2 / sto-3g (HF start):
1. QP energies are differentiable w.r.t. the mean-field inputs (mo_energy,
   mo_coeff) through the implicit custom VJP, matching central finite
   differences.
2. The implicit and unrolled differentiation modes agree.

These tests are the acceptance gate for the "differentiable GW" design
goal; they must pass before any GW stage is considered complete.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import g0w0_cd_restricted

_NW = 40


def _h2_inputs():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    assert mf.converged
    res = mf.scf_result
    inputs = mf._scf_inputs
    df_factors = getattr(inputs, "df_factors", None)
    if df_factors is None:
        from gradscf.df import eri_pair_matrix_to_df_factors

        df_factors = eri_pair_matrix_to_df_factors(
            inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
        )
    return dict(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=1,
        df_factors=jnp.asarray(df_factors),
        fock_matrix=jnp.asarray(res.fock_matrix),
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        density_matrix=jnp.asarray(res.density_matrix),
    )


def _homo_qp(inputs, mo_energy, mo_coeff, diff_mode="implicit"):
    res = g0w0_cd_restricted(
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=inputs["nocc"],
        df_factors=inputs["df_factors"],
        fock_matrix=inputs["fock_matrix"],
        hcore_matrix=inputs["hcore_matrix"],
        density_matrix=inputs["density_matrix"],
        nw=_NW,
        diff_mode=diff_mode,
    )
    return res.mo_energy[inputs["nocc"] - 1]


def _finite_diff_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        grad[idx] = (fn(jnp.asarray(xp)) - fn(jnp.asarray(xm))) / (2.0 * eps)
    return grad


def test_qp_energy_grad_wrt_mo_energy_matches_finite_diff():
    inputs = _h2_inputs()
    fn = lambda e: float(_homo_qp(inputs, e, inputs["mo_coeff"]))
    grad_ad = np.asarray(jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"]))(inputs["mo_energy"]))
    grad_fd = _finite_diff_grad(fn, inputs["mo_energy"])
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-6)


def test_qp_energy_grad_wrt_mo_coeff_matches_finite_diff():
    inputs = _h2_inputs()
    fn = lambda c: float(_homo_qp(inputs, inputs["mo_energy"], c))
    grad_ad = np.asarray(jax.grad(lambda c: _homo_qp(inputs, inputs["mo_energy"], c))(inputs["mo_coeff"]))
    grad_fd = _finite_diff_grad(fn, inputs["mo_coeff"])
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-5)


def test_qp_energy_implicit_matches_unrolled():
    inputs = _h2_inputs()
    grad_implicit = jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"], "implicit"))(
        inputs["mo_energy"]
    )
    grad_unrolled = jax.grad(lambda e: _homo_qp(inputs, e, inputs["mo_coeff"], "unrolled"))(
        inputs["mo_energy"]
    )
    np.testing.assert_allclose(grad_implicit, grad_unrolled, rtol=1e-5, atol=1e-7)
