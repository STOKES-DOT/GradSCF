"""Frozen real RKS/UKS orbitals and invariant, differentiable grid features.

Preparing a spectral reference is an eager offline operation. Once supplied,
its coefficient/energy leaves and the raw AO grid retain their own AD paths;
this module does not assert a differentiated Fock-to-eigenvector map.
"""
from dataclasses import replace
from math import isfinite
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import array_device, tree_storage_bytes, workspace_bytes
from gradscf.integrals.grids.blocks import AOGrid, ao_point_bytes
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from gradscf.solvers.eigen.spectral import inverse_sqrt
from gradscf.tools.features import _restricted_rho_kernel, _spin_tau
from .orbital import OrbitalReference
from .types import Target, _check


def prepare_orbital_reference(target, fock, *, method, max_memory=8000.):
    """Freeze the complete real RKS/UKS spectrum of a declared AO Fock matrix.

    ``method`` records the supplied operator's provenance. Occupations are
    integer Aufbau populations from ``target.nelec``; the target may remain a
    correlated density different from the reference density. Internal orbital
    degeneracies are allowed. No occupied/virtual gap is imposed here.

    The shared inverse square root and bounded dense Hermitian solver supply
    the entire spectrum. Raw Ritz arrays are used only after checking their
    convergence, presence and finiteness. JIT/AD of this offline preparation is
    explicitly unsupported; supplying already prepared references to
    :func:`orbital_grid_features` has a separate differentiable contract.
    ``max_memory`` is the total array-planning budget in decimal MB.
    """
    if any(isinstance(value, jax.core.Tracer)
           for value in jax.tree.leaves((target, fock))):
        raise NotImplementedError('Orbital reference preparation is frozen/offline; Fock-map AD/JIT is unsupported')
    if not isinstance(target, Target):
        raise TypeError('Orbital preparation requires an iKS Target')
    if target.layout not in ('rks', 'uks'):
        raise NotImplementedError('Orbital preparation currently requires real RKS/UKS references')
    if not isinstance(method, str) or not method:
        raise ValueError('Declare the source orbital reference method')
    fock = jnp.asarray(fock)
    if fock.shape != target.density.shape or jnp.iscomplexobj(fock):
        raise ValueError('Fock must be real and match the target AO/spin shape')
    if jnp.iscomplexobj(target.overlap):
        raise ValueError('Orbital preparation requires a real overlap matrix')
    dtype = jnp.result_type(fock, target.overlap)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError('Fock and overlap require real floating-point data')
    n = target.overlap.shape[0]
    counts = (target.nelec//2,) if target.layout == 'rks' else target.nelec
    if any(number > n for number in counts):
        raise ValueError('Target electron counts exceed the complete orbital space')
    config = EigenSolverConfig(method='dense', nroots=n)
    if n > config.max_dense:
        raise ValueError('Orbital preparation exceeds the bounded dense spectrum dimension')
    # Charge spectral validation, whitening, both spin spectra, and retained
    # references before numerical work; the solver budgets its own scratch.
    resident = tree_storage_bytes(target, fock)
    workspace_bytes(max_memory, resident_bytes=resident
                    + (16*len(counts)*n*n+8*len(counts)*n)*np.dtype(dtype).itemsize,
                    device=array_device(fock, target.overlap))
    fock = fock.astype(dtype)
    scale = jnp.maximum(1., jnp.linalg.norm(fock))
    # Restored fixed operators already satisfy the iKS entrywise 1e-10
    # contract. A Frobenius-only epsilon test would spuriously tighten it
    # with AO dimension; discard only accepted antisymmetric roundoff.
    tolerance = jnp.maximum(1e-10,32*jnp.finfo(dtype).eps*scale)
    _check(jnp.all(jnp.isfinite(fock))
           & (jnp.max(jnp.abs(fock-fock.swapaxes(-1, -2))) <= tolerance),
           'Fock must be finite and symmetric')
    fock=.5*(fock+fock.swapaxes(-1,-2))
    _check(target.valid, 'Orbital preparation requires a valid target')
    _check(jnp.min(jnp.linalg.eigvalsh(target.overlap)) > 1e-12,
           'Overlap must be positive definite without inverse-square-root clipping')
    orthogonalizer = inverse_sqrt(target.overlap.astype(dtype), 1e-12)
    blocks = fock if target.layout == 'uks' else fock[None]
    whitened = orthogonalizer@blocks@orthogonalizer
    whitened = .5*(whitened+whitened.swapaxes(-1, -2))
    coefficients, energies = [], []
    for matrix in whitened:
        solved = solve_hermitian(matrix, config=config,
            response=EigenResponseConfig(target='subspace'), max_memory=max_memory,
            reserved_bytes=tree_storage_bytes(target, fock, orthogonalizer,
                                              whitened, coefficients, energies))
        if (solved.raw_values.shape != (n,) or solved.raw_vectors.shape != (n, n)
                or solved.raw_residual_norms.shape != (n,)
                or solved.raw_present.shape != (n,)):
            raise ValueError('Orbital preparation requires a complete raw Ritz spectrum')
        _check(jnp.all(solved.converged) & jnp.all(solved.raw_present)
               & jnp.all(solved.response_valid)
               & jnp.all(jnp.isfinite(solved.raw_values))
               & jnp.all(jnp.isfinite(solved.raw_vectors))
               & jnp.all(jnp.isfinite(solved.raw_residual_norms))
               & jnp.all(solved.raw_residual_norms <= config.atol),
               'Orbital preparation requires a converged finite complete raw Ritz spectrum')
        coefficients.append(orthogonalizer@solved.raw_vectors)
        energies.append(solved.raw_values)
    coefficients, energies = jnp.stack(coefficients), jnp.stack(energies)
    maximum = 2 if target.layout == 'rks' else 1
    occupations = jnp.stack([maximum*(jnp.arange(n) < number) for number in counts])
    if target.layout == 'rks':
        coefficients, energies, occupations = coefficients[0], energies[0], occupations[0]
    return OrbitalReference(target, coefficients, occupations, energies, method=method)


def _feature_grid_plan(reference, grid, transform, windows, max_memory):
    dtype = np.dtype(jnp.result_type(reference.coefficients, reference.energies,
        reference.occupations, grid.coords, grid.centers, grid.exponents,
        grid.coefficients, *(() if transform is None else (transform,))))
    n, m = grid.nao, reference.coefficients.shape[-1]
    spins = 2 if reference.target.layout == 'uks' else 1
    count = len(windows)
    fields = 2+count
    # Transported coefficients, weighted products, retained AO matrices and
    # their assembly temporaries; no grid-by-orbital storage is constructed.
    matrix_bytes = dtype.itemsize*(3*spins*n*m+(3*spins+2)*(1+count)*n*n
                                   +4*(1+count)*spins*m)
    # map retains padded scan outputs as well as the trimmed scalar fields.
    output_bytes = 2*grid.ngrids*fields*dtype.itemsize
    available = workspace_bytes(max_memory,
        resident_bytes=tree_storage_bytes(reference, grid, transform)+matrix_bytes+output_bytes,
        device=array_device(reference.coefficients, grid.exponents))
    per_point = ao_point_bytes(grid, deriv=1)+dtype.itemsize*(4*n+3*fields+8)
    width = min(grid.block_size, available//per_point)
    if width < 1:
        raise ValueError('Orbital feature memory budget cannot hold one AO quadrature point')
    return replace(grid, block_size=int(width)), dtype


@jax.jit
def _orbital_grid_program(grid, occupied_density, virtual_densities):
    def fields(ao, derivative, laplacian, coordinates, weights):
        values = {'rho_occ': _restricted_rho_kernel(ao, occupied_density[None]),
                  'tau_occ': _spin_tau(derivative, occupied_density)}
        for index in range(virtual_densities.shape[0]):
            values[f'virtual_{index}'] = _restricted_rho_kernel(
                ao, virtual_densities[index:index+1])
        return values
    return grid.map(fields, deriv=1)


def orbital_grid_features(reference, grid, *, transform=None, windows=(.2, 1.),
                          max_memory=8000.):
    """Return spin-summed scalar orbital fields on a raw Cartesian AOGrid.

    ``rho_occ`` and ``tau_occ`` use the reference's occupied AO density, with
    tau = 1/2 sum_i f_i |grad(phi_i)|^2. Each ``virtual_k`` uses virtual weights
    ``maximum * eta**2 / ((epsilon-mu)**2+eta**2)``: maximum is 2 for RKS and 1
    per UKS spin, eta is ``windows[k]`` in Hartree, and each spin's mu is the
    midpoint of its highest occupied and lowest virtual energy. No energy
    labels are needed. Phase, consistent orbital permutations and rotations
    within equal-energy/equal-occupation subspaces leave these fields invariant.

    Occupations must be integer. Nonempty windows require occupied and virtual
    orbitals in every spin block. ``transform`` has shape (naocart, naosph) and
    transports the reference coefficients to the raw grid's AO representation.
    Coefficient, energy, transform, basis and geometry derivatives are retained
    for supplied references. There is no differentiated offline preparation or
    implicit state response in this function. Only small scalar outputs are
    retained; AO values/derivatives are produced by bounded, checkpointed blocks.
    """
    if not isinstance(reference, OrbitalReference):
        raise TypeError('Orbital features require a declared OrbitalReference')
    if reference.target.layout not in ('rks', 'uks'):
        raise NotImplementedError('Orbital grid features currently require real RKS/UKS references')
    if not isinstance(grid, AOGrid):
        raise TypeError('Orbital features require a raw AOGrid')
    if any(jnp.iscomplexobj(value) for value in (reference.coefficients,
            reference.target.overlap, grid.coords, grid.weights, grid.centers,
            grid.exponents, grid.coefficients)):
        raise ValueError('Orbital features require real references and a real AO grid')
    try:
        windows = tuple(windows)
    except TypeError as error:
        raise ValueError('Energy windows require positive finite widths in Hartree') from error
    if any(isinstance(eta, (bool, np.bool_)) or not isinstance(eta, Real)
           or not isfinite(eta) or eta <= 0 for eta in windows):
        raise ValueError('Energy window widths must be positive finite real numbers in Hartree')
    windows = tuple(float(eta) for eta in windows)
    if transform is not None:
        transform = jnp.asarray(transform)
        if (transform.shape != (grid.nao, reference.coefficients.shape[-2])
                or jnp.iscomplexobj(transform)):
            raise ValueError('Orbital transform requires real (naocart,naosph) coefficients')
    elif grid.nao != reference.coefficients.shape[-2]:
        raise ValueError('Grid and reference AO dimensions differ; supply an orbital transform')
    grid, dtype = _feature_grid_plan(reference, grid, transform, windows, max_memory)
    if transform is not None:
        _check(jnp.all(jnp.isfinite(transform)), 'Orbital transform must be finite')
    reference_valid = reference.valid
    integer_occupations = reference.integer_occupations
    _check(reference_valid, 'Orbital grid features require a valid orbital reference')
    _check(integer_occupations, 'Orbital grid features require integer occupations')
    maximum = 2 if reference.target.layout == 'rks' else 1
    occupied = reference.occupations > maximum/2
    valid = reference_valid & integer_occupations
    if windows:
        split = jnp.all(jnp.any(occupied, axis=-1) & jnp.any(~occupied, axis=-1))
        _check(split, 'Energy windows require an occupied/virtual split in each spin block')
        valid &= split
    coefficients = reference.coefficients.astype(dtype)
    if transform is not None:
        coefficients = transform.astype(dtype)@coefficients
    occupations = reference.occupations.astype(dtype)
    energies = reference.energies.astype(dtype)
    if reference.target.layout == 'rks':
        coefficients, occupations, energies, occupied = (
            value[None] for value in (coefficients, occupations, energies, occupied))
    transpose = coefficients.swapaxes(-1, -2)
    occupied_density = jnp.sum((coefficients*occupations[:, None, :])@transpose, axis=0)
    virtual_densities = []
    if windows:
        homo = jnp.max(jnp.where(occupied, energies, -jnp.inf), axis=-1)
        lumo = jnp.min(jnp.where(~occupied, energies, jnp.inf), axis=-1)
        mu = .5*(homo+lumo)
        for eta in windows:
            weight = maximum*(~occupied)*eta**2/((energies-mu[:, None])**2+eta**2)
            virtual_densities.append(jnp.sum(
                (coefficients*weight[:, None, :])@transpose, axis=0))
    virtual_densities = (jnp.stack(virtual_densities) if windows else
                         jnp.zeros((0, grid.nao, grid.nao), dtype=dtype))
    # Concrete invalid inputs were rejected above; invalid traced charts remain
    # visible in their numerical outputs rather than bypassing the contract.
    scale = jnp.where(valid, 1., jnp.nan)
    return _orbital_grid_program(grid, occupied_density*scale, virtual_densities*scale)


__all__ = ['prepare_orbital_reference', 'orbital_grid_features']
