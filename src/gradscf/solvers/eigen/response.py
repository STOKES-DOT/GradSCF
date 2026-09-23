"""Common first-order horizontal spectral response; rank one is a single state."""

from typing import Literal
import jax
import jax.numpy as jnp
from ..linear.implicit import checked_linear_solve

EigenGradientMode = Literal["eigenvalue_only", "implicit_eigenvector"]


def attach_spectral_response(apply, basis, *, config, valid=True):
    """Solve Q(AZ-ZH)=-Q(dA)X, X.T Z=0 in a fixed Euclidean metric.

    For rank one this is the isolated eigenvector differential. At higher rank
    this private horizontal frame may only feed basis-invariant observables.
    The rank-k Sylvester extension never divides by internal eigenvalue gaps.
    """
    x = jax.lax.stop_gradient(basis)
    n, k = x.shape
    ax = apply(x)
    h = jax.lax.stop_gradient(0.5 * (x.T @ ax + ax.T @ x))
    project = lambda z: x @ (x.T @ z)
    complement = lambda z: z - project(z)

    def sylvester(v):
        z = v.reshape(n, k)
        qz = complement(z)
        physical = (complement(apply(qz) - qz @ h) + project(z)).reshape(-1)
        return jnp.where(valid, physical, v)

    rhs = -complement(ax - jax.lax.stop_gradient(ax))
    correction, _ = checked_linear_solve(sylvester, rhs.reshape(-1), config=config)
    return x + correction.reshape(n, k)
