"""Fixed inverse-reference evaluation for analytic or learned scalar energies."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf.dft.derivatives import xc_energy_and_potential_from_density
from .thermal import metric, orthogonal_gauge
from .orbital import OrbitalReference
from .response import orbital_response, density_norm2


class Evaluation(NamedTuple):
    total_energy: object
    energy_error: object
    operator_mse: object
    stationarity_mse: object
    fock: object
    valid: object
    rotation_norm2: object = None
    predicted_density_norm2: object = None
    reference_density_norm2: object = None
    xc_potential_mse: object = None

    @property
    def commutator_mse(self):
        """The legacy stationarity_mse is a commutator, not a rotation loss."""
        return self.stationarity_mse


class XCReferenceEvaluation(NamedTuple):
    """Explicit XC-energy assessment with a label separate from total energy."""
    xc_energy: object
    error: object
    valid: object


def evaluate_xc_energy(reference, xc_energy_fn, params=None, *, target_xc_energy=None):
    """Evaluate an XC-energy callback against an explicitly supplied XC label.

    ``Target.energy`` intentionally remains a total-energy field. This helper
    keeps wavefunction-derived ``E_xc`` labels separate and performs no density
    derivative, Fock construction, SCF, or inverse-map operation.
    """
    if target_xc_energy is None:
        raise ValueError('An explicit XC-energy label is required')
    value = jnp.asarray(xc_energy_fn(params, reference.density))
    label = jnp.asarray(target_xc_energy)
    if value.shape != () or label.shape != () or jnp.iscomplexobj(value) or jnp.iscomplexobj(label):
        raise ValueError('XC energy and label must be real scalar values')
    valid = reference.valid & jnp.isfinite(value) & jnp.isfinite(label)
    for leaf in jax.tree.leaves(params):
        valid &= jnp.all(jnp.isfinite(leaf))
    return XCReferenceEvaluation(value, value-label, valid)


def xc_energy_loss(reference, xc_energy_fn, params=None, *, target_xc_energy=None,
                   weights=None, scale=1.):
    """Return a dimensionless explicit-XC loss and its raw assessment."""
    weights = {'mse': 1.} if weights is None else weights
    if not weights or set(weights)-{'mse', 'mae'}:
        raise ValueError('XC-energy weights must contain mse and/or mae')
    if any(float(weight) < 0. or not bool(jnp.isfinite(weight)) for weight in weights.values()):
        raise ValueError('XC-energy weights must be finite and nonnegative')
    scale = float(scale)
    if not bool(jnp.isfinite(scale)) or scale <= 0.:
        raise ValueError('XC-energy scale must be positive and finite in Hartree')
    score = evaluate_xc_energy(reference, xc_energy_fn, params,
                               target_xc_energy=target_xc_energy)
    residual = score.error / scale
    value = jnp.asarray(0., dtype=residual.dtype)
    if 'mse' in weights:
        value = value + float(weights['mse']) * residual**2
    if 'mae' in weights:
        value = value + float(weights['mae']) * jnp.abs(residual)
    return jnp.where(score.valid, value, jnp.nan), score


def evaluate(reference, energy_fn, params=None, *, quantities=None, grid=None, gap_floor=.05,
             xc_energy_fn=None):
    """Evaluate E(params,D), its physical AO derivative and reference residuals.

    energy_fn returns TOTAL energy in Hartree. operator_mse is a gauge-aligned
    orthonormal-frame Fock mismatch; stationarity_mse is a commutator residual
    only, not a ground-state/stability or self-consistent density certificate.
    All model-parameter derivatives, including those of the operator loss,
    remain live. No candidate SCF or implicit inverse-map gradient is performed.
    New orbital references default to energy only. For legacy thermal results,
    omitted quantities preserves energy/operator/commutator evaluation. Select
    energy, operator, commutator, stationarity or predicted_density explicitly.
    Stationarity returns a squared rotation norm, not a mean or SCF certificate.
    """
    target = reference.target
    if quantities is None:
        quantities = ('energy',) if isinstance(reference, OrbitalReference) else ('energy', 'operator', 'commutator')
    requested = frozenset(quantities)
    if not requested or requested-{'energy', 'operator', 'commutator', 'stationarity', 'predicted_density', 'xc_potential'}:
        raise ValueError('Unknown fixed-reference assessment quantity')
    needs_fock = bool(requested-{'energy', 'xc_potential'})
    if needs_fock:
        derivative = xc_energy_and_potential_from_density(params, molecule=None, density=reference.density,
            xc_energy_fn=lambda p, _, d: energy_fn(p, d), check_finite=False)
        value, fock = derivative.xc_energy, derivative.vxc_matrix
    else:
        value, fock = jnp.asarray(energy_fn(params, reference.density)), None
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError('energy_fn must return a real scalar total energy')
    valid = reference.valid & jnp.isfinite(value)
    for leaf in jax.tree.leaves(params):
        valid &= jnp.all(jnp.isfinite(leaf))
    operator_mse = commutator_mse = rotation = predicted = baseline = xc_mse = None
    if needs_fock:
        valid &= jnp.all(jnp.isfinite(fock))
    if requested & {'operator', 'commutator'}:
        _, x, root = metric(target)
        if 'operator' in requested:
            error = orthogonal_gauge(fock-reference.fock, x)
            operator_mse = jnp.mean(jnp.abs(error)**2)
            valid &= jnp.isfinite(operator_mse)
        if 'commutator' in requested:
            density = root@reference.density@root
            physical = x@fock@x
            commutator = physical@density-density@physical
            commutator_mse = jnp.mean(jnp.abs(commutator)**2)
            valid &= jnp.isfinite(commutator_mse)
    if requested & {'stationarity', 'predicted_density'}:
        delta, rotation, chart_valid = orbital_response(reference, fock, gap_floor,
            with_density='predicted_density' in requested)
        valid &= chart_valid & jnp.isfinite(rotation)
        if 'predicted_density' in requested:
            predicted, baseline = density_norm2(reference, delta, grid)
            valid &= jnp.isfinite(predicted) & jnp.isfinite(baseline)
    if 'xc_potential' in requested:
        teacher = getattr(reference, 'xc_potential', None)
        if xc_energy_fn is None or teacher is None:
            raise ValueError('XC-potential assessment requires an explicit xc_energy_fn and AO XC teacher')
        derivative = xc_energy_and_potential_from_density(params, molecule=None, density=reference.density,
            xc_energy_fn=lambda p, _, d: xc_energy_fn(p, d), check_finite=False)
        xc_value, potential = derivative.xc_energy, derivative.vxc_matrix
        if jnp.shape(xc_value) != () or jnp.iscomplexobj(xc_value):
            raise ValueError('xc_energy_fn must return a real scalar XC energy')
        _, x, _ = metric(target)
        xc_mse = jnp.mean(jnp.abs(orthogonal_gauge(potential-teacher, x))**2)
        valid &= jnp.isfinite(xc_value) & jnp.isfinite(xc_mse)
    energy_error = None if target.energy is None else value-target.energy
    return Evaluation(value, energy_error, operator_mse, commutator_mse, fock, valid,
                      rotation, predicted, baseline, xc_mse)


def energy_function(molecule, functional, *, layout='uks', nuclear_repulsion=None, component='total'):
    """Bind a common TOTAL-energy callback for fixed-reference assessment.

    Named traditional recipes reuse the existing SCF energy assembly. Custom
    scalar XC and built-in semilocal neural models reuse their own live-density
    evaluation, adding h+J exactly once. Legacy frozen-HFX/PT2 neural channels
    need a separately tested live-energy contract and are rejected here.
    """
    from gradscf.dft.functional import Functional, custom_xc
    from gradscf.dft.xc import TraditionalXCFunctional, hybrid_coeff, xc_type
    from gradscf.scf.differentiable import _replace_molecule
    from gradscf.scf.uks import UKSConfig, _spin_jk_matrices, _raw_fock_and_energy_for_state
    from gradscf.scf.energy import XCContribution, unrestricted_energy
    from .types import _check

    if component not in ('total', 'xc'):
        raise ValueError('Energy component must be total or xc')
    enuc = getattr(molecule, 'nuclear_repulsion', None) if nuclear_repulsion is None else nuclear_repulsion
    if enuc is None and component == 'xc': enuc = 0.
    if enuc is None:
        raise ValueError('Supply nuclear_repulsion explicitly when the state has no nuclear-energy metadata')
    enuc = jnp.asarray(enuc)
    if enuc.shape != () or jnp.iscomplexobj(enuc):
        raise ValueError('nuclear_repulsion must be a real scalar')
    _check(jnp.isfinite(enuc), 'nuclear_repulsion must be finite')
    if isinstance(functional, str): functional = custom_xc(functional)
    named = isinstance(functional, TraditionalXCFunctional)
    if not named and not (isinstance(functional, Functional) or callable(getattr(functional, 'energy_from_molecule', None))):
        raise TypeError('Supply a traditional recipe, custom_xc Functional or an energy-bearing neural model')
    if not named and (getattr(functional, 'include_hfx_channel', False) or getattr(functional, 'include_pt2_channel', False)):
        raise NotImplementedError('Frozen neural HFX/PT2 channels do not define the live energy required here')
    if layout not in ('rks', 'uks', 'gks'): raise ValueError('Use rks, uks or gks layout')
    if layout == 'gks':
        if not named: raise NotImplementedError('The current shared custom spinor energy adapter is not implemented')
        if component == 'xc': raise NotImplementedError('A separate spinor XC energy adapter is not implemented')
        from gradscf.scf.gks import GKSConfig, generalized_energy_and_fock
        def spinor(params, density):
            return generalized_energy_and_fock(density=density, hcore=molecule.h1e,
                eri=molecule.rep_tensor, nuclear_repulsion=enuc,
                ao=molecule.ao, ao_deriv1=molecule.ao_deriv1, grid_weights=molecule.grid.weights,
                config=GKSConfig(xc_spec=functional.spec))[0]
        return spinor
    def xc(params, density):
        spin = density if layout == 'uks' else jnp.stack((density/2, density/2))
        state = _replace_molecule(molecule, rdm1=spin)
        return (functional.energy_for_density(params, state, spin) if isinstance(functional, Functional)
                else functional.energy_from_molecule(params, state))
    if component == 'xc' and not named:
        return xc
    if component == 'xc' and getattr(molecule, 'dispersion_energy', None) is not None:
        raise NotImplementedError('Separate non-XC dispersion energy before constructing an XC teacher')
    cfg = UKSConfig(xc_spec=functional.spec) if named else None
    rep = getattr(molecule, 'eri_pair_matrix', None)
    if rep is None: rep = molecule.rep_tensor
    df = getattr(molecule, 'df_factors', None)
    direct = getattr(molecule, 'direct_basis', None)
    def total(params, density):
        spin = density if layout == 'uks' else jnp.stack((density/2, density/2))
        state = _replace_molecule(molecule, rdm1=spin)
        if named:
            value = _raw_fock_and_energy_for_state(density_a=spin[0], density_b=spin[1],
                ao=state.ao, ao_deriv1=state.ao_deriv1, weights=state.grid.weights,
                h=state.h1e, eri=rep, df_factors=df, enuc=enuc,
                alpha=hybrid_coeff(functional.spec), cfg=cfg, xc_kind=xc_type(functional.spec),
                direct_basis=direct, grid_ao=getattr(state, 'grid_ao', None),
                eri_lr=getattr(state, 'eri_lr', None), dispersion_energy=getattr(state, 'dispersion_energy', None),
                compute_potential=False)[0]
            if component == 'total': return value
            _, coulomb, ka, kb = _spin_jk_matrices(rep, spin[0], spin[1], df_factors=df,
                                                 direct_basis=direct, with_k=False)
            return value-unrestricted_energy(spin, state.h1e, coulomb, jnp.stack((ka, kb)),
                XCContribution(jnp.asarray(0., value.dtype), 0., 0.), nuclear_repulsion=enuc)
        _, coulomb, ka, kb = _spin_jk_matrices(rep, spin[0], spin[1], df_factors=df,
                                             direct_basis=direct, with_k=False)
        exc = xc(params, density)
        return unrestricted_energy(spin, state.h1e, coulomb, jnp.stack((ka, kb)),
            XCContribution(exc, 0., 0.), nuclear_repulsion=enuc)
    return total


def hybrid_xc_energy_function(molecule, functional, *, layout='rks',
                              exact_exchange_fraction=.25):
    """Bind a parameterized hybrid XC energy for fixed-reference assessment.

    ``functional`` supplies the live semilocal XC energy through the existing
    ``Functional.energy_for_density`` contract.  The exact-exchange term is
    evaluated from the supplied AO density and the molecule's existing ERI,
    density-fitting, or direct-SCF representation.  This callback returns only
    ``E_xc`` in Hartree; it does not provide an SCF binding, a hybrid response
    protocol, or an inverse-GKS solver.

    The restricted layout is represented internally by equal alpha/beta
    densities, matching the existing unrestricted energy assembly.  GKS is
    intentionally rejected because the shared custom spinor energy adapter is
    not implemented.
    """
    from gradscf.dft.functional import Functional
    from gradscf.scf.differentiable import _replace_molecule
    from gradscf.scf.uks import _spin_jk_matrices

    if not isinstance(functional, Functional):
        raise TypeError('hybrid_xc_energy_function requires a custom Functional')
    if layout not in ('rks', 'uks'):
        raise ValueError('Use RKS or UKS layout; custom spinor hybrid energy is not implemented')
    fraction = float(exact_exchange_fraction)
    if not jnp.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('exact_exchange_fraction must be finite and in [0, 1]')

    eri = getattr(molecule, 'eri_pair_matrix', None)
    if eri is None:
        eri = getattr(molecule, 'rep_tensor', None)
    df_factors = getattr(molecule, 'df_factors', None)
    direct_basis = getattr(molecule, 'direct_basis', None)
    if eri is None and df_factors is None and direct_basis is None:
        raise ValueError('Fixed-reference hybrid energy requires AO ERI, DF factors, or direct basis')

    def xc_energy(params, density):
        spin = density if layout == 'uks' else jnp.stack((density / 2, density / 2))
        state = _replace_molecule(molecule, rdm1=spin)
        semilocal = functional.energy_for_density(params, state, spin)
        _, _, exchange_a, exchange_b = _spin_jk_matrices(
            eri, spin[0], spin[1], df_factors=df_factors, with_k=True,
            direct_basis=direct_basis,
        )
        exchange = jnp.stack((exchange_a, exchange_b))
        exact_exchange = -.5 * fraction * jnp.einsum(
            'sij,sij->', spin, exchange, precision=jax.lax.Precision.HIGHEST
        )
        return jnp.asarray(semilocal) + exact_exchange

    return xc_energy
