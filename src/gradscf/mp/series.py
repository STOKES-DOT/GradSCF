"""MP of arbitrary finite order from a projected residual and Taylor lifting.

The state uses intermediate normalization. The kth wavefunction has excitation
rank at most 2k for a two-body Hamiltonian. A Rayleigh quotient of the state
through k gives exact energy coefficients through 2k+1 (Wigner's rule).
No FCI product space, dense Hamiltonian or eigensolve is constructed here.
Retained determinants/connections still grow combinatorially; this is a bounded
in-core generic engine, not the streaming DF-MP2 algorithm.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet

from ..ci.space import make_ci_space, make_uci_space
from ..ci import hamiltonian
from ..fci.cistring import excite
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear
from .types import MPResult
from .integrals import _reference_data, canonical_error, _active_orbitals


def make_series_space(nmo, nocc, *, frozen=None, config):
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    build = make_uci_space if isinstance(nocc, tuple) else make_ci_space
    return build(nmo, nocc, max_excitation=2*k, frozen=frozen,
                 max_determinants=config.max_determinants)


def _jet(fun, primals, coefficients):
    return jet(fun, primals, coefficients, factorial_scaled=False)


def _amplitudes(first, space, unrestricted):
    n = space.nmo
    ns = space.nocc if unrestricted else (space.nocc, space.nocc)
    fr = space.frozen if unrestricted else (space.frozen, space.frozen)
    selections = [_active_orbitals(n, no, f) for no, f in zip(ns, fr)]
    lookup = {d: i for i, d in enumerate(space.determinants)}
    def block(a, b):
        oa, va = selections[a]
        ob, vb = selections[b]
        indices, phases = [], []
        for i in oa:
            for j in ob:
                for x in va:
                    for y in vb:
                        det, sign = excite(space.determinants[0], (a*n+int(i), b*n+int(j)),
                                           (a*n+int(x), b*n+int(y)))
                        indices.append(lookup[det] if sign else 0)
                        phases.append(sign)
        shape = (len(oa), len(ob), len(va), len(vb))
        return (first[jnp.asarray(indices, dtype=jnp.int32)] * jnp.asarray(phases)).reshape(shape)
    return (block(0, 0), block(0, 1), block(1, 1)) if unrestricted else block(0, 1)


def run_series(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config):
    nuclear_repulsion = jnp.asarray(nuclear_repulsion)
    if nuclear_repulsion.shape != () or jnp.iscomplexobj(nuclear_repulsion):
        raise ValueError("nuclear_repulsion must be a real scalar")
    hs, gs, focks, reference_energy = _reference_data(h1, eri, nocc)
    unrestricted = isinstance(nocc, tuple)
    space = make_series_space(hs[0].shape[0], nocc, frozen=frozen, config=config)
    op = hamiltonian.build_hamiltonian(h1, eri, space)
    n = space.nmo
    ff = focks if unrestricted else (focks[0], focks[0])
    reference_bits = space.determinants[0]
    # Changes in occupations, including both spin frames, define F_N directly.
    changes = np.array([[(int(d & (1 << p) != 0)-int(reference_bits & (1 << p) != 0))
                          for p in range(2*n)] for d in space.determinants])
    gaps = jnp.asarray(changes, dtype=op.diagonal.dtype) @ jnp.concatenate([jnp.diag(f) for f in ff])
    diagonal0 = reference_energy + gaps
    phi = jnp.zeros(space.size, op.diagonal.dtype).at[0].set(1.)
    def fluctuation(c):
        return op(c)-diagonal0*c
    def residual(y, lam):
        c = jnp.concatenate((jnp.ones(1, y.dtype), y[:-1]))
        w = fluctuation(c)
        return jnp.concatenate((gaps[1:]*y[:-1]-y[-1]*y[:-1]+lam*w[1:],
                                (lam*w[0]-y[-1])[None]))
    zero = jnp.zeros(space.size, op.diagonal.dtype)
    lam0 = jnp.zeros((), zero.dtype)
    _, jacobian = jax.linearize(lambda y: residual(y, lam0), zero)
    jacdiag = jnp.concatenate((gaps[1:], -jnp.ones(1, zero.dtype)))
    safe = jnp.where(jnp.abs(jacdiag) > config.denominator_tol, jacdiag, 1.)
    operator = LinearOperator((space.size, space.size), zero.dtype, jacobian, diagonal=jacdiag)
    linear = LinearSolverConfig(rtol=1e-10, atol=1e-12, maxiter=10, restart=min(20, space.size))
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    ys, waves = [], [phi]
    solved, norm = jnp.asarray(True), lam0
    for degree in range(1, k+1):
        inputs = (tuple(ys+[zero]), tuple([jnp.ones_like(lam0)]+[lam0]*(degree-1)))
        _, terms = _jet(residual, (zero, lam0), inputs)
        response = solve_linear(operator, -terms[-1], config=linear,
            preconditioner=lambda v: v/safe, transpose_preconditioner=lambda v: v/safe)
        ys.append(response.solution)
        waves.append(jnp.concatenate((jnp.zeros(1, zero.dtype), response.solution[:-1])))
        solved = solved & response.converged
        norm = jnp.maximum(norm, response.residual_norm)
    def rayleigh(c, lam):
        hc = diagonal0*c+lam*fluctuation(c)
        return jnp.vdot(c, hc)/jnp.vdot(c, c)
    # Wigner 2k+1: omitted higher state coefficients cannot affect these energies.
    coefficients = tuple(waves[1:]+[jnp.zeros_like(phi)]*max(0, config.order-k))[:config.order]
    _, es = _jet(rayleigh, (phi, lam0),
        (coefficients, tuple([jnp.ones_like(lam0)]+[lam0]*(config.order-1))))
    corrections = jnp.stack(es[1:])
    first = waves[1]
    # Project first-order channels, then apply the same generic energy functional.
    alpha_rank = np.array([((d ^ reference_bits) & ((1 << n)-1)).bit_count()//2
                           for d in space.determinants])
    beta_rank = np.array([((d ^ reference_bits) >> n).bit_count()//2 for d in space.determinants])
    def channel(mask):
        c1 = jnp.where(jnp.asarray(mask), first, 0.)
        _, terms = _jet(rayleigh, (phi, lam0), ((c1, jnp.zeros_like(phi)), (jnp.ones_like(lam0), lam0)))
        return terms[1]
    ss = channel(((alpha_rank == 2) & (beta_rank == 0)) | ((alpha_rank == 0) & (beta_rank == 2)))
    os = channel((alpha_rank == 1) & (beta_rank == 1))
    minimum = jnp.min(jnp.abs(gaps[1:]), initial=jnp.inf)
    error = canonical_error(focks)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in (*hs, *gs, *focks)]))
    valid = (finite & jnp.isfinite(nuclear_repulsion) & (error < config.canonical_tol)
             & (minimum > config.denominator_tol) & solved & jnp.all(jnp.isfinite(corrections)))
    # Multiplicative guard rejects invalid JVP and VJP as well as primal values.
    mask = lambda x: x*jnp.where(valid, jnp.ones_like(x), jnp.full_like(x, jnp.nan))
    corr = jnp.sum(corrections)
    e3 = corrections[1] if config.order >= 3 else jnp.zeros_like(corr)
    t2 = jax.tree.map(mask, _amplitudes(first, space, unrestricted)) if config.with_t2 else None
    return MPResult(mask(reference_energy+nuclear_repulsion+corr), mask(corr),
        reference_energy+nuclear_repulsion, mask(corrections[0]), mask(e3), mask(ss), mask(os), t2,
        minimum, error, valid, mask(corrections),
        mask(jnp.stack(waves)) if config.with_coefficients else None, norm)
