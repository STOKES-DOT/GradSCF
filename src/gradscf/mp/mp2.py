"""Canonical MP2 with occupied-index streaming and ordinary JAX AD."""
import jax
import jax.numpy as jnp

from .types import MPConfig, MPResult
from .integrals import prepare_integrals


def _channel(left_energy, right_energy, no, nj, *, block=None, factors=None,
             same_spin=False, restricted=False, config):
    nv, nb = left_energy.size - no, right_energy.size - nj
    dtype = jnp.result_type(left_energy, right_energy)
    amplitudes = jnp.zeros((no, nj, nv, nb), dtype) if config.with_t2 else None
    initial = (jnp.zeros((), dtype), jnp.zeros((), dtype),
               jnp.asarray(jnp.inf, dtype), amplitudes)
    if no == 0 or nj == 0 or nv == 0 or nb == 0:
        return initial
    de = right_energy[:nj, None, None] - right_energy[None, None, nj:]
    def step(i, state):
        ss, os, minimum, ts = state
        direct = (block[i].transpose(1, 0, 2) if block is not None else
                  jnp.einsum("Qa,Qjb->jab", factors[0][:, i], factors[1],
                             precision="highest"))
        denominator = left_energy[i] - left_energy[None, no:, None] + de
        allowed = jnp.ones(denominator.shape, dtype=bool)
        if same_spin:
            allowed = ((jnp.arange(nj)[:, None, None] != i)
                       & (jnp.arange(nv)[None, :, None] != jnp.arange(nb)[None, None, :]))
        minimum = jnp.minimum(minimum, jnp.min(jnp.where(allowed, jnp.abs(denominator), jnp.inf)))
        safe = jnp.where(allowed & (jnp.abs(denominator) > config.denominator_tol), denominator, 1.)
        numerator = direct - direct.swapaxes(1, 2) if same_spin else direct
        t = jnp.where(allowed, numerator / safe, 0.)
        if restricted:
            os = os + jnp.sum(t * direct)
            ss = ss + jnp.sum(t * (direct - direct.swapaxes(1, 2)))
        elif same_spin:
            ss = ss + .25 * jnp.sum(t * numerator)
        else:
            os = os + jnp.sum(t * direct)
        return ss, os, minimum, ts.at[i].set(t) if ts is not None else None
    return jax.lax.fori_loop(0, no, jax.checkpoint(step, prevent_cse=False), initial)


def evaluate_mp2(ints, config):
    restricted = len(ints.nocc) == 1
    pairs = ((0, 0),) if restricted else ((0, 0), (0, 1), (1, 1))
    channels = []
    for k, (a, b) in enumerate(pairs):
        channels.append(_channel(ints.energies[a], ints.energies[b], ints.nocc[a], ints.nocc[b],
            block=ints.blocks[k] if ints.blocks is not None else None,
            factors=(ints.factors[a], ints.factors[b]) if ints.factors is not None else None,
            same_spin=not restricted and a == b, restricted=restricted, config=config))
    ss = sum(row[0] for row in channels)
    os = sum(row[1] for row in channels)
    minimum = jnp.min(jnp.stack([row[2] for row in channels]))
    e2 = ss + os
    valid = (ints.finite & (ints.canonical_error < config.canonical_tol)
             & (minimum > config.denominator_tol) & jnp.isfinite(e2))
    mask = lambda x: jnp.where(valid, x, jnp.nan)
    ts = tuple(mask(row[3]) if row[3] is not None else None for row in channels)
    t2 = ts[0] if restricted else (ts if config.with_t2 else None)
    return MPResult(mask(ints.reference_energy + e2), mask(e2), ints.reference_energy,
                    mask(e2), jnp.zeros_like(e2), mask(ss), mask(os), t2,
                    minimum, ints.canonical_error, valid)


def run_mp(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config=None):
    """Pure MO-array entry, differentiable in h1, ERIs and nuclear repulsion.

    nocc is a static integer (RHF) or (nalpha, nbeta) tuple (UHF).
    Eager facades additionally verify source convergence and freshness.
    """
    cfg = MPConfig() if config is None else config
    if cfg.order == 3:
        from .mp3 import evaluate_mp3
        return evaluate_mp3(h1, eri, nocc=nocc, frozen=frozen,
                            nuclear_repulsion=nuclear_repulsion, config=cfg)
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen,
                             nuclear_repulsion=nuclear_repulsion)
    return evaluate_mp2(ints, cfg)
