"""Bounded executable reuse, separate from numerical reference preparation.

Only immutable driver controls enter the cache key. Reference arrays and
external model parameters are ordinary dynamic arguments, never cache values.
Opaque/mutable callbacks and shell-direct providers retain their existing
execution path rather than being captured as static physical state.
"""
from dataclasses import fields
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from ..scf.molecules import RestrictedMolecule, UnrestrictedMolecule
from .response_options import normalize_response_kernel_options
from ._semilocal_response import SemilocalResponseFunctional
from ._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional
from ..dft.functional import Functional
from .validation import finite_checks


def _structural_metadata(value):
    if value is None or isinstance(value, (str, bool, int, float, complex, np.generic)):
        return True
    return type(value) in (tuple, frozenset) and all(_structural_metadata(x) for x in value)


def _cacheable(driver):
    molecule = driver.molecule
    if type(molecule) not in (RestrictedMolecule, UnrestrictedMolecule):
        return False
    occupations = (('nocc',) if type(molecule) is RestrictedMolecule else
                   ('nocc_alpha', 'nocc_beta'))
    if any(getattr(molecule, name) is None for name in occupations):
        return False
    if getattr(molecule, 'direct_basis', None) is not None:
        # Preparing a shell-direct action outside tracing can select a bounded
        # exact ERI cache. Do not change that algorithm by tracing its provider.
        return False
    try:
        metadata = molecule.tree_flatten()[1]
        hash(metadata)
    except TypeError:
        # Legacy references may keep mutable, opaque producer metadata. Do
        # not reinterpret or discard that state to force a cache hit.
        return False
    if not _structural_metadata(metadata):
        # Identity-hashed producer objects can hide mutable physical data or
        # callbacks. Retain their eager path instead of caching a closure.
        return False
    functional = driver.xc_functional
    if functional is not None:
        # Unknown frozen objects can still hide physical arrays or spatial
        # HF callbacks. Hashability alone is not a dynamic-input contract.
        if type(functional) not in (SemilocalResponseFunctional,
                                    UnrestrictedSemilocalResponseFunctional, Functional):
            return False
        try:
            hash(functional)
        except TypeError:
            return False
    return True


@lru_cache(maxsize=32)
def _program(driver_type, controls, implementation, nstates, device, signature):
    options = dict(controls)

    @jax.jit
    def solve(molecule, xc_params):
        # Tracers do not expose a concrete device. Keep the known execution
        # device in the trace's planning context and program specialization.
        with jax.default_device(device):
            with finite_checks() as checks:
                current = driver_type(molecule=molecule, xc_params=xc_params, **options)
                result = getattr(current, implementation)(nstates)
            valid = jnp.stack(checks) if checks else jnp.ones((0,), dtype=bool)
            return result, valid

    return solve


def execute_response(driver, nstates, implementation):
    """Run the unchanged solver with a stable same-signature JIT boundary."""
    if not _cacheable(driver):
        return getattr(driver, implementation)(nstates)
    leaves, tree = jax.tree_util.tree_flatten((driver.molecule, driver.xc_params))
    device = None
    for value in leaves:
        current = getattr(value, 'device', None)
        current = current() if callable(current) else current
        if hasattr(current, 'platform'):
            device = current
            break
    device = device or jax.config.jax_default_device
    if device is None and not any(isinstance(x, jax.core.Tracer) for x in leaves):
        # Uncommitted NumPy-only inputs use JAX's ordinary default execution
        # target. Merely reading ndarray.device == 'cpu' does not establish it.
        device = jax.devices()[0]
    if device is None:
        # An outer transform with no explicit target retains the old path;
        # do not guess a GPU's budget from traced CPU/host operands.
        return getattr(driver, implementation)(nstates)
    controls = {}
    for field in fields(driver):
        if field.name not in ('molecule', 'xc_params'):
            controls[field.name] = getattr(driver, field.name)
    controls['response_kernel_options'] = normalize_response_kernel_options(
        controls['response_kernel_options'], default_max_memory=driver.molecule.max_memory)
    try:
        avals = tuple(jax.typeof(x) for x in leaves)
        signature = (tree, tuple((tuple(aval.shape), str(aval.dtype),
                                 bool(getattr(aval, 'weak_type', False))) for aval in avals))
        hash(signature)
    except TypeError:
        return getattr(driver, implementation)(nstates)
    result, valid = _program(type(driver), tuple(controls.items()), implementation, nstates, device, signature)(
        driver.molecule, driver.xc_params)
    if not isinstance(valid, jax.core.Tracer) and not bool(np.all(np.asarray(valid))):
        raise ValueError('AO fock_matrix or prepared response kernel contains NaN or Inf.')
    return result
