"""Restricted CI and spin-adapted CIS solvers."""
import jax.numpy as jnp

from ..solvers import LinearOperator, EigenSolverConfig, solve_hermitian
from .hamiltonian import build_hamiltonian, validate_integrals
from .space import frozen_indices
from .types import CIConfig, CIResult, CISResult


def _eigenpairs(apply, diagonal, config):
    dim = diagonal.size
    operator = LinearOperator((dim, dim), diagonal.dtype,
        lambda v: apply(v[:, None])[:, 0], diagonal=diagonal, matmat=apply)
    shared = EigenSolverConfig(method=config.solver, nroots=config.nroots,
        atol=config.conv_tol, maxiter=config.max_cycle, max_subspace=config.max_space,
        gradient_mode=config.gradient_mode, adjoint_tol=config.adjoint_tol,
        adjoint_maxiter=config.adjoint_max_cycle)
    result = solve_hermitian(operator, config=shared)
    return result.values, result.vectors, result.residual_norms, result.converged


def solve_ci(h1, eri, space, *, nuclear_repulsion=0.0, config=None):
    """Variational CI in a static M_s=0 space, with first-order energy AD.

    Frozen-core energy is retained because frozen electrons remain in the
    determinants. Derivatives assume converged, isolated roots and fixed topology.
    """
    config = CIConfig() if config is None else config
    op = build_hamiltonian(h1, eri, space)
    energies, vectors, residuals, converged = _eigenpairs(op, op.diagonal, config)
    reference = op.diagonal[0]
    return CIResult(energies + nuclear_repulsion, energies - reference,
                    reference + nuclear_repulsion, vectors, residuals, converged)


def restricted_fock(h1, eri, nocc):
    return h1 + 2 * jnp.einsum("pqii->pq", eri[:, :, :nocc, :nocc]) - jnp.einsum(
        "piiq->pq", eri[:, :nocc, :nocc, :])


def solve_cis(h1, eri, *, nocc, singlet=True, frozen=None, config=None):
    """Spin-adapted HF singles with unit-normalized spatial amplitudes.

    A(ia,jb) = delta(ij) F(ab) - delta(ab) F(ij)
               + 2 (ia|jb) [singlet only] - (ij|ab).
    The occupied-virtual Fock block must vanish (stationary HF reference).
    CIS reference: Foresman et al. (1992), doi:10.1021/j100180a030.
    See ci/REFERENCES.md for the separate numerical-response references.
    """
    config = CIConfig() if config is None else config
    h1, eri = validate_integrals(h1, eri)
    nmo = h1.shape[0]
    if not isinstance(nocc, int) or nocc < 1 or nocc >= nmo:
        raise ValueError("CIS requires occupied and virtual orbitals")
    frozen = frozen_indices(nmo, nocc, frozen)
    occ = jnp.asarray([i for i in range(nocc) if i not in frozen], dtype=jnp.int32)
    vir = jnp.asarray([i for i in range(nocc, nmo) if i not in frozen], dtype=jnp.int32)
    no, nv = occ.size, vir.size
    if no * nv == 0:
        raise ValueError("CIS has no active single excitations")
    fock = restricted_fock(h1, eri, nocc)
    f_oo, f_vv = fock[jnp.ix_(occ, occ)], fock[jnp.ix_(vir, vir)]
    exchange = eri[jnp.ix_(occ, occ, vir, vir)]
    coulomb = eri[jnp.ix_(occ, vir, occ, vir)]

    def apply(vectors):
        x = vectors.reshape(no, nv, -1)
        y = jnp.einsum("ab,ibk->iak", f_vv, x) - jnp.einsum("ij,jak->iak", f_oo, x)
        y = y - jnp.einsum("ijab,jbk->iak", exchange, x)
        if singlet:
            y = y + 2 * jnp.einsum("iajb,jbk->iak", coulomb, x)
        return y.reshape(no * nv, -1)

    diagonal = (jnp.diag(f_vv)[None, :] - jnp.diag(f_oo)[:, None]
                - jnp.einsum("iiaa->ia", exchange))
    if singlet:
        diagonal = diagonal + 2 * jnp.einsum("iaia->ia", coulomb)
    energies, vectors, residuals, converged = _eigenpairs(apply, diagonal.ravel(), config)
    return CISResult(energies, vectors.T.reshape(config.nroots, no, nv), residuals, converged,
                     config.gradient_mode == "implicit_eigenvector", singlet)
