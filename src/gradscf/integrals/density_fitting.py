"""Auxiliary-basis RI from native 2c/3c integrals, never from full ERIs.

The native integral calls are forward-only. At fixed centers/exponents the
primitive RI factors may be cached; projection through contraction matrices
is fully JAX differentiable, including zero raw contraction coefficients.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from .basis import BasisTopology, BasisParameters
from .plan import make_plan


@dataclass(frozen=True)
class AuxiliaryPlan:
    orbital: BasisTopology
    auxiliary: BasisTopology
    combined: object

    def evaluate(self, parameters, auxiliary_parameters):
        from .backends.native_compact import evaluate

        p, a = parameters, auxiliary_parameters
        combined = BasisParameters(
            p.exponents + a.exponents,
            p.coefficients + a.coefficients,
            jnp.concatenate((p.centers, a.centers)),
            p.nuclear_coords,
        )
        atm, bas, env = self.combined.pack(combined)
        n, m = self.orbital.nao, self.auxiliary.nao
        kwargs = dict(cart=self.orbital.cart, split=len(self.orbital.angular_momenta))
        metric = evaluate(atm, bas, env, (m, m), layout=2, **kwargs)
        three = evaluate(atm, bas, env, (m, n * (n + 1) // 2), layout=3, **kwargs)
        return metric, three

    def factors(self, parameters, auxiliary_parameters, *, lindep=1e-12):
        """Whiten packed three-center integrals by the auxiliary Coulomb metric.

        A host Cholesky factorization is used for this fixed-basis cache. If
        the auxiliary metric is numerically dependent, discard eigenmodes
        below ``lindep``. This rank choice is not a basis/geometry AD rule.
        """
        from scipy.linalg import cholesky, solve_triangular, eigh

        metric, three = self.evaluate(parameters, auxiliary_parameters)
        m = np.asarray(metric)
        a = np.asarray(three)
        if not np.isfinite(lindep) or lindep <= 0:
            raise ValueError("lindep must be positive.")
        try:
            lower = cholesky(m, lower=True)
            factors = solve_triangular(lower, a, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            values, vectors = eigh(m)
            keep = values > lindep
            if not np.any(keep):
                raise ValueError(
                    "Auxiliary Coulomb metric has no positive independent modes."
                )
            factors = (vectors[:, keep] / np.sqrt(values[keep])).T @ a
        return jnp.asarray(factors)


def make_auxiliary_plan(orbital, auxiliary):
    if orbital.cart != auxiliary.cart:
        raise ValueError("Orbital and auxiliary angular representations must match.")
    top = BasisTopology(
        orbital.angular_momenta + auxiliary.angular_momenta,
        orbital.primitive_counts + auxiliary.primitive_counts,
        orbital.contraction_counts + auxiliary.contraction_counts,
        orbital.nuclear_charges,
        orbital.cart,
    )
    return AuxiliaryPlan(orbital, auxiliary, make_plan(top))


def unpack_factors(packed, nao):
    """Unpack only a three-index tensor, never a four-index ERI."""
    packed = jnp.asarray(packed)
    if packed.ndim != 2 or packed.shape[1] != nao * (nao + 1) // 2:
        raise ValueError("Packed RI factor shape mismatch.")
    i, j = np.tril_indices(nao)
    out = jnp.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
    out = out.at[:, i, j].set(packed)
    return out.at[:, j, i].set(packed)


def project_factors(packed_primitive, transform, *, block_size=64):
    """B_c[Q] = T.T B_p[Q] T, unpacking at most block_size auxiliary rows."""
    t = jnp.asarray(transform)
    factors = jnp.asarray(packed_primitive)
    rank = factors.shape[0]
    width = min(int(block_size), rank)
    if rank == 0:
        return jnp.zeros((0, t.shape[1], t.shape[1]), dtype=t.dtype)
    if width < 1:
        raise ValueError("block_size must be positive.")

    def block(f):
        return jnp.einsum(
            "pi,Qpq,qj->Qij",
            t,
            unpack_factors(f, t.shape[0]),
            t,
            precision=jax.lax.Precision.HIGHEST,
        )

    out = jnp.zeros((rank, t.shape[1], t.shape[1]), dtype=jnp.result_type(t, factors))

    def step(i, out):
        b = jax.lax.dynamic_slice_in_dim(factors, i * width, width, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(out, block(b), i * width, axis=0)

    out = jax.lax.fori_loop(0, rank // width, step, out)
    start = rank // width * width
    if start < rank:
        out = out.at[start:].set(block(factors[start:]))
    return out


__all__ = ["AuxiliaryPlan", "make_auxiliary_plan", "unpack_factors", "project_factors"]
