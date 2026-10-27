"""Fixed inverse-reference evaluation for analytic or learned scalar energies."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .thermal import metric, orthogonal_gauge


class Evaluation(NamedTuple):
    total_energy: object
    energy_error: object
    operator_mse: object
    stationarity_mse: object
    fock: object
    valid: object


def evaluate(reference, energy_fn, params=None):
    """Evaluate E(params,D), its physical AO derivative and reference residuals.

    energy_fn returns TOTAL energy in Hartree. operator_mse is a gauge-aligned
    orthonormal-frame Fock mismatch; stationarity_mse is a commutator residual
    only, not a ground-state/stability or self-consistent density certificate.
    All model-parameter derivatives, including those of the operator loss,
    remain live. No candidate SCF or implicit inverse-map gradient is performed.
    """
    target = reference.target
    value, derivative = jax.value_and_grad(energy_fn, argnums=1)(params, reference.density)
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError('energy_fn must return a real scalar total energy')
    fock = derivative.conj()
    fock = .5*(fock+fock.conj().swapaxes(-1, -2))
    _, x, root = metric(target)
    error = orthogonal_gauge(fock-reference.fock, x)
    density = root@reference.density@root
    physical = x@fock@x
    commutator = physical@density-density@physical
    operator_mse = jnp.mean(jnp.abs(error)**2)
    stationarity = jnp.mean(jnp.abs(commutator)**2)
    valid = (reference.valid & jnp.isfinite(value) & jnp.all(jnp.isfinite(fock))
             & jnp.isfinite(operator_mse) & jnp.isfinite(stationarity))
    energy_error = None if target.energy is None else value-target.energy
    return Evaluation(value, energy_error, operator_mse, stationarity, fock, valid)


def energy_function(molecule, functional, *, layout='uks', nuclear_repulsion=None):
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

    enuc = getattr(molecule, 'nuclear_repulsion', None) if nuclear_repulsion is None else nuclear_repulsion
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
        from gradscf.scf.gks import GKSConfig, generalized_energy_and_fock
        def spinor(params, density):
            return generalized_energy_and_fock(density=density, hcore=molecule.h1e,
                eri=molecule.rep_tensor, nuclear_repulsion=enuc,
                ao=molecule.ao, ao_deriv1=molecule.ao_deriv1, grid_weights=molecule.grid.weights,
                config=GKSConfig(xc_spec=functional.spec))[0]
        return spinor
    cfg = UKSConfig(xc_spec=functional.spec) if named else None
    rep = getattr(molecule, 'eri_pair_matrix', None)
    if rep is None: rep = molecule.rep_tensor
    df = getattr(molecule, 'df_factors', None)
    direct = getattr(molecule, 'direct_basis', None)
    def total(params, density):
        spin = density if layout == 'uks' else jnp.stack((density/2, density/2))
        state = _replace_molecule(molecule, rdm1=spin)
        if named:
            return _raw_fock_and_energy_for_state(density_a=spin[0], density_b=spin[1],
                ao=state.ao, ao_deriv1=state.ao_deriv1, weights=state.grid.weights,
                h=state.h1e, eri=rep, df_factors=df, enuc=enuc,
                alpha=hybrid_coeff(functional.spec), cfg=cfg, xc_kind=xc_type(functional.spec),
                direct_basis=direct, grid_ao=getattr(state, 'grid_ao', None),
                eri_lr=getattr(state, 'eri_lr', None), dispersion_energy=getattr(state, 'dispersion_energy', None),
                compute_potential=False)[0]
        _, coulomb, ka, kb = _spin_jk_matrices(rep, spin[0], spin[1], df_factors=df,
                                             direct_basis=direct, with_k=False)
        if isinstance(functional, Functional):
            exc = functional.energy_for_density(params, state, spin)
        else:
            exc = functional.energy_from_molecule(params, state)
        return unrestricted_energy(spin, state.h1e, coulomb, jnp.stack((ka, kb)),
            XCContribution(exc, 0., 0.), nuclear_repulsion=enuc)
    return total
