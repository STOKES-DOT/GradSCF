"""One frozen Fock evaluation and Aufbau solve per candidate; no candidate SCF."""
from math import isfinite
from typing import NamedTuple

import jax.numpy as jnp

from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .assessment import evaluate
from .orbital import OrbitalReference
from .response import density_norm2
from .types import _check


class Reward(NamedTuple):
    """Raw single-sample diagnostics and a dimensionless reward in [0,1].

    Energies/gaps are Hartree. Density errors are relative grid L2 norms,
    resolving UKS spin channels or GKS charge and magnetization components.
    Invalid candidates receive value=0, loss=NaN, valid=False. Raw operators
    and diagnostics are retained; no successful replacement model is supplied.
    """
    value: object
    loss: object
    variational_gap: object
    wu_yang_gap: object
    density_error: object
    reference_density_error: object
    total_energy: object
    density: object
    fock: object
    gap: object
    valid: object


def _aufbau(fock, nocc, gap_tolerance):
    """Use the shared isolated-subspace response, including complex spinors.

    Realification doubles a complex Hermitian problem and its occupied rank.
    It preserves complete degenerate pairs and reuses the real solver's AD.
    """
    n = fock.shape[0]
    if nocc == 0:
        return jnp.zeros_like(fock), jnp.asarray(jnp.inf), jnp.asarray(True)
    if nocc == n:
        return jnp.eye(n, dtype=fock.dtype), jnp.asarray(jnp.inf), jnp.asarray(True)
    complex_problem = jnp.iscomplexobj(fock)
    matrix = (jnp.block([[fock.real, -fock.imag], [fock.imag, fock.real]])
              if complex_problem else fock)
    rank = 2*nocc if complex_problem else nocc
    result = solve_hermitian(matrix,
        config=EigenSolverConfig(method='dense', nroots=rank, atol=1e-11),
        response=EigenResponseConfig(target='subspace', gap_atol=gap_tolerance, gap_rtol=0.),
        probes=jnp.eye(matrix.shape[0], dtype=matrix.dtype))
    projector = result.projection
    if complex_problem:
        projector = projector[:n, :n]+1j*projector[n:, :n]
    return projector, result.boundary_gap, result.response_valid


def reward(reference, energy_fn, params=None, *, grid, energy_scale=.01,
           density_scale=.01, weights=None, gap_tolerance=1e-8):
    """Assess a candidate TOTAL energy at one fixed orbital reference.

    First prepare an integer-occupation, gapped Aufbau OrbitalReference offline
    (e.g. with invert(method='wu_yang')). This function differentiates the
    candidate E(params,D0) once to obtain F, fills its lowest orbitals once,
    and compares D1 with the correlated target. F is never updated at D1.
    A callback from energy_function shares traditional/custom/neural XC with
    DFT; a direct callback must include kinetic, external, Hartree and XC terms.

    Let Es(F)=Tr(F D1). The frozen-operator variational excess is
    G=Tr(F D0)-Es(F)>=0. The signed Wu-Yang difference is
    W(F0)-W(F)=Tr(F0 D0)-Es(F)-Tr((F0-F) Dtarget).
    Only with a common fixed local base and an exact local inverse is the
    latter a nonnegative Wu-Yang dual gap. Finite inverse errors can make it
    negative; for nonlocal GKS it is an operator diagnostic, not a local dual.

    loss = w_gap*max(G,0)/energy_scale
         + w_density*(density_error/density_scale)**2; value = 1/(1+loss).
    Weights default to one each. energy_scale is Ha, density_scale is a
    dimensionless relative L2 error. Both are workflow choices, not physical
    accuracy guarantees. The inverse baseline is reported without subtraction.
    No total-energy label is required. Add energy objectives separately when
    needed: density-only scores cannot identify additive energy constants.

    JIT and model-parameter AD use complete occupied-subspace invariants;
    internal degeneracy is allowed. A closing occupied/virtual gap is invalid.
    This single-step diagnostic does not certify candidate SCF convergence,
    stability, or the error of its final self-consistent density.
    """
    if not isinstance(reference, OrbitalReference):
        raise TypeError('reward requires an offline OrbitalReference')
    for name, value in (('energy_scale', energy_scale), ('density_scale', density_scale),
                        ('gap_tolerance', gap_tolerance)):
        if not isfinite(float(value)) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    weights = {'gap': 1., 'density': 1.} if weights is None else weights
    if not weights or set(weights)-{'gap', 'density'}:
        raise ValueError('Reward weights must select gap and/or density')
    if any(not isfinite(float(w)) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('Reward weights must be finite, nonnegative and not all zero')
    _check(reference.valid, 'Invalid orbital reference')
    _check(reference.integer_occupations, 'Reward requires integer occupations')
    maximum = 2 if reference.target.layout == 'rks' else 1
    occupied = reference.occupations > maximum/2
    lowest_virtual = jnp.min(jnp.where(occupied, jnp.inf, reference.energies), axis=-1)
    highest_occupied = jnp.max(jnp.where(occupied, reference.energies, -jnp.inf), axis=-1)
    reference_good = reference.valid & reference.integer_occupations & jnp.all(
        lowest_virtual-highest_occupied > gap_tolerance)
    _check(reference_good, 'Reference must be a gapped Aufbau ground state')

    assessment = evaluate(reference, energy_fn, params, quantities=('energy', 'fock'))
    fock = assessment.fock
    # The complete reference orbitals already whiten S; no new S diagonalization.
    c = reference.coefficients
    adjoint = lambda a: a.conj().swapaxes(-1, -2)
    orthogonal = adjoint(c)@fock@c
    counts = (reference.target.nelec if reference.target.layout == 'uks' else
              (reference.target.nelec//maximum,))
    blocks = orthogonal if reference.target.layout == 'uks' else orthogonal[None]
    projectors, gaps, accepted = [], [], []
    for block, count in zip(blocks, counts):
        projector, gap, valid = _aufbau(block, count, gap_tolerance)
        projectors.append(maximum*projector)
        gaps.append(gap)
        accepted.append(valid)
    density = c@jnp.stack(projectors)@adjoint(c)
    if reference.target.layout != 'uks':
        density = density[0]
    trace = lambda a, d: jnp.einsum('...ij,...ji->', a, d).real
    spectral_energy = trace(fock, density)
    excess = trace(fock, reference.density)-spectral_energy
    wy_gap = trace(reference.fock, reference.density)-spectral_energy-trace(
        reference.fock-fock, reference.target.density)
    error2, baseline2 = density_norm2(reference, density-reference.density, grid, relative=True)
    # Keep the exact-zero branch smooth: maximum(0,G) would halve AD curvature
    # at G=0. Only suppress negative roundoff, retaining the raw diagnostic.
    nonnegative_excess = jnp.where(excess < 0., 0., excess)
    loss = (float(weights.get('gap', 0.))*nonnegative_excess/energy_scale
            +float(weights.get('density', 0.))*error2/density_scale**2)
    valid = (reference_good & assessment.valid & jnp.all(jnp.stack(accepted))
             & jnp.isfinite(loss) & jnp.isfinite(error2) & jnp.isfinite(baseline2))
    return Reward(jnp.where(valid, 1/(1+loss), 0.), jnp.where(valid, loss, jnp.nan),
                  excess, wy_gap, jnp.sqrt(error2), jnp.sqrt(baseline2),
                  assessment.total_energy, density, fock, jnp.min(jnp.stack(gaps)), valid)
