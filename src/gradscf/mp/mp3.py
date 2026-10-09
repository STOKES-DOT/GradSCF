"""Restricted MP3 from the first-order connected doubles interaction.

For canonical HF, T1^(1)=0 and E^(3) depends only on T2^(2).
The interaction action is the zero-amplitude derivative of the existing CCSD
doubles residual with its one-body Fock part removed. No CC iteration, fitted
scaling, or finite-difference derivative is used. See REFERENCES.md.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp

from ..cc.integrals import prepare_integrals, denominators
from ..cc.rccsd import residual, correlation_energy
from .mp2 import evaluate_mp2
from .integrals import prepare_integrals as prepare_mp_integrals


def evaluate_mp3(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config):
    if isinstance(nocc, tuple):
        raise NotImplementedError("MP3 currently requires a restricted canonical HF reference")
    mp_ints = prepare_mp_integrals(h1, eri, nocc=nocc, frozen=frozen,
                                   nuclear_repulsion=nuclear_repulsion)
    first = evaluate_mp2(mp_ints, replace(config, order=2, with_t2=True))
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen,
                             nuclear_repulsion=nuclear_repulsion)
    _, d2 = denominators(ints)
    # F_N belongs to H0, so retain only the fluctuation interaction in this action.
    interaction = ints._replace(fock=jnp.zeros_like(ints.fock))
    zero1 = jnp.zeros((ints.nocc, ints.nvir), dtype=ints.fock.dtype)
    zero2 = jnp.zeros_like(first.t2)
    action = lambda t: residual(zero1, t, interaction, model="ccd")[1]
    _, linear = jax.jvp(action, (zero2,), (first.t2,))
    second = linear / jnp.where(jnp.abs(d2) > config.denominator_tol, d2, 1.)
    e3 = correlation_energy(zero1, second, interaction, model="lccd")
    valid = first.valid & jnp.isfinite(e3)
    e3 = jnp.where(valid, e3, jnp.nan)
    ecorr = first.e2 + e3
    return first._replace(total_energy=first.reference_energy + ecorr,
                           correlation_energy=ecorr, e3=e3, valid=valid,
                           t2=first.t2 if config.with_t2 else None)
