"""Gaussian normalization; raw parsing is host-only, parameter math uses JAX."""
import math
import numpy as np
import jax.numpy as jnp

def _gaussian_int(n: int, alpha: np.ndarray) -> np.ndarray:
    """Match PySCF's radial Gaussian integral helper."""

    n1 = 0.5 * (n + 1)
    return math.gamma(n1) / (2.0 * np.asarray(alpha, dtype=float) ** n1)


def _gto_norm(l: int, exponents: np.ndarray) -> np.ndarray:
    """Match pyscf.gto.mole.gto_norm for radial factors."""

    return 1.0 / np.sqrt(_gaussian_int(2 * l + 2, 2.0 * np.asarray(exponents, dtype=float)))


def _normalize_raw_shell_coefficients(
    l: int,
    primitive_rows: list[list[object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw PySCF basis rows to mol.bas_exp / mol.bas_ctr_coeff convention."""

    rows = sorted(
        [[float(value) for value in row] for row in primitive_rows],
        reverse=True,
    )
    exponents = np.asarray([row[0] for row in rows], dtype=float)
    coeff_rows = np.asarray([row[1:] for row in rows], dtype=float)
    if coeff_rows.ndim == 1:
        coeff_rows = coeff_rows[:, None]

    # PySCF make_env:
    #   cs = raw_coeff * gto_norm(l, es)
    #   cs = _nomalize_contracted_ao(l, es, cs)
    # PySCF bas_ctr_coeff:
    #   bas_ctr_coeff = cs / gto_norm(l, es)
    # Therefore the public coefficients used by mol.bas_ctr_coeff are:
    #   raw_coeff * shell_normalization
    primitive_norm = coeff_rows * _gto_norm(l, exponents)[:, None]
    metric = _gaussian_int(2 * l + 2, exponents[:, None] + exponents[None, :])
    shell_norm = 1.0 / np.sqrt(
        np.einsum("pi,pq,qi->i", primitive_norm, metric, primitive_norm)
    )
    coefficients = coeff_rows * shell_norm[None, :]
    return exponents, coefficients



def radial_primitive_norm(l, exponents):
    """PySCF radial primitive convention, differentiated w.r.t. exponents."""
    a = jnp.asarray(exponents)
    n = l + 1.5
    return 1 / jnp.sqrt(math.gamma(n) / (2 * (2*a)**n))


def normalized_shell_coefficients(l, exponents, coefficients):
    """Raw (nprim,nctr) coefficients -> normalized libcint environment values."""
    a, c = jnp.asarray(exponents), jnp.asarray(coefficients)
    weighted = c * radial_primitive_norm(l, a)[:, None]
    metric = math.gamma(l + 1.5) / (2 * (a[:, None] + a[None, :])**(l + 1.5))
    norm2 = jnp.einsum("pi,pq,qi->i", weighted, metric, weighted)
    return weighted / jnp.sqrt(norm2)[None, :]

__all__ = ["radial_primitive_norm", "normalized_shell_coefficients"]
