from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..data.molecule import MoleculeSpec


_EIGH_DEGENERACY_TOL = 1e-6
_EIGH_BROADENING = 1e-10


@jax.custom_jvp
def _safe_symmetric_eigh(matrix: Array) -> tuple[Array, Array]:
    values, vectors = jnp.linalg.eigh(matrix)
    return values, vectors


@_safe_symmetric_eigh.defjvp
def _safe_symmetric_eigh_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    values, vectors = _safe_symmetric_eigh(matrix)
    value_diff = values[None, :] - values[:, None]
    nondegenerate = jnp.abs(value_diff) >= _EIGH_DEGENERACY_TOL
    # Avoid singular arithmetic even in an inactive branch of the JVP. This
    # matters when the rule itself is differentiated for force supervision.
    regular_gap = 1.0 / jnp.where(nondegenerate, value_diff, 1.0)
    broadened_gap = value_diff / (value_diff * value_diff + _EIGH_BROADENING)
    response = jnp.where(nondegenerate, regular_gap, broadened_gap)
    response = response.at[jnp.diag_indices_from(response)].set(0.0)
    inner = vectors.T @ (.5*(tangent+tangent.T)) @ vectors
    return (values, vectors), (jnp.diag(inner), vectors @ (response*inner))


def _contains_jax_tracer(value: Any) -> bool:
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, MoleculeSpec):
        return _contains_jax_tracer((value.coords_bohr, value.charges))
    if isinstance(value, dict):
        return any(_contains_jax_tracer(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jax_tracer(item) for item in value)
    leaves = jax.tree_util.tree_leaves(value)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _host_float_unless_traced(value: Any) -> Any:
    return value if _contains_jax_tracer(value) else float(value)


def _validate_density_matrix(
    density: Array | None,
    *,
    nao: int,
    dtype: Any,
    label: str,
    method: str,
) -> Array | None:
    if density is None:
        return None
    dm = jnp.asarray(density, dtype=dtype)
    if dm.ndim != 2 or tuple(dm.shape) != (nao, nao):
        raise ValueError(f"{label} must be a square ({nao}, {nao}) matrix for {method}.")
    return 0.5 * (dm + dm.T)


def _spectral_orthogonalizer(overlap: Array, eps: float) -> Array:
    eigvals, eigvecs = _safe_symmetric_eigh(overlap)
    clipped = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(clipped ** -0.5) @ eigvecs.T


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _orthogonalizer(overlap: Array, eps: float) -> Array:
    return _spectral_orthogonalizer(overlap, eps)


@_orthogonalizer.defjvp
def _orthogonalizer_jvp(eps, primals, tangents):
    (overlap,), (tangent,) = primals, tangents
    inverse_root = _orthogonalizer(overlap, eps)

    def positive_definite_response(_):
        # Differentiate the matrix function, not its individual eigenvectors:
        # sqrt(S) dX + dX sqrt(S) = -X dS X, X = S**(-1/2).
        # The Sylvester operator stays nonsingular at repeated eigenvalues.
        root = overlap @ inverse_root
        root = .5 * (root + root.T)
        rhs = -inverse_root @ (.5 * (tangent + tangent.T)) @ inverse_root

        def solve(_, value):
            values, vectors = jnp.linalg.eigh(root)
            inner = vectors.T @ value @ vectors
            return vectors @ (inner / (values[:, None] + values[None, :])) @ vectors.T

        return jax.lax.custom_linear_solve(
            lambda value: root @ value + value @ root,
            rhs, solve=solve, symmetric=True,
        )

    # Preserve the existing regularized response when eigenvalue clipping is
    # active. Exact smooth matrix-function derivatives apply above the cutoff.
    response = jax.lax.cond(
        jnp.min(jnp.linalg.eigvalsh(overlap)) > eps,
        positive_definite_response,
        lambda _: jax.jvp(lambda s: _spectral_orthogonalizer(s, eps),
                          (overlap,), (tangent,))[1],
        operand=None,
    )
    return inverse_root, response


def _diagonalize_fock(
    fock: Array,
    x: Array,
    eigenvalue_jitter: float = 0.0,
) -> tuple[Array, Array]:
    f_ortho = x.T @ fock @ x
    f_ortho = 0.5 * (f_ortho + f_ortho.T)
    if eigenvalue_jitter != 0.0:
        shift = jnp.arange(f_ortho.shape[0], dtype=f_ortho.dtype) * eigenvalue_jitter
        f_ortho = f_ortho + jnp.diag(shift)
    mo_energy, coeff_ortho = _safe_symmetric_eigh(f_ortho)
    mo_coeff = x @ coeff_ortho
    return mo_energy, mo_coeff


def _build_density_from_occ(mo_coeff: Array, mo_occ: Array) -> Array:
    occ = jnp.asarray(mo_occ, dtype=mo_coeff.dtype)
    return jnp.einsum("pi,i,qi->pq", mo_coeff, occ, mo_coeff, precision=Precision.HIGHEST)
