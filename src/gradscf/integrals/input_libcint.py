"""Host integral caching and coordinate-aware one-electron assembly."""

from __future__ import annotations

from typing import Any, Callable
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from .autodiff import bind_libcint_integral_constant, libcint_int1e_with_coords
from .backends.pyscf_mol import build_libcint_mol, libcint_intor_name
from .backends.gpu4pyscf import gpu4pyscf_int2e_full, gpu4pyscf_int2e_s4
from .input_cache import _cache_bounded

_LIBCINT_HOST_INTEGRAL_CACHE_MAXSIZE = 64
_LIBCINT_HOST_INTEGRAL_CACHE: dict[tuple[Any, ...], Any] = {}

def _cached_libcint_host_integral(
    *,
    mol: Any,
    integral_name: str,
    geometry_anchor: Array,
    geometry_grad_policy: str,
    loader: Any,
) -> Array:
    key = (id(mol), str(integral_name), str(geometry_grad_policy))
    cached = _LIBCINT_HOST_INTEGRAL_CACHE.get(key)
    if cached is not None:
        return cached
    value = bind_libcint_integral_constant(
        loader(),
        geometry_anchor=geometry_anchor,
        integral_name=integral_name,
        geometry_grad_policy=geometry_grad_policy,
    )
    _cache_bounded(
        _LIBCINT_HOST_INTEGRAL_CACHE,
        key,
        value,
        _LIBCINT_HOST_INTEGRAL_CACHE_MAXSIZE,
    )
    return value


def _gpu4pyscf_eri(
    *,
    packed: bool,
    atom: Any,
    basis: Any,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    mol_kwargs: dict[str, Any],
) -> Array:
    mol = build_libcint_mol(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=int(charge),
        spin=int(spin),
        cart=bool(cart),
        verbose=int(verbose),
        **mol_kwargs,
    )
    integral_fn = gpu4pyscf_int2e_s4 if packed else gpu4pyscf_int2e_full
    return jnp.asarray(integral_fn(mol))


def _libcint_one_electron_with_coords(
    *,
    coords_bohr: Array,
    symbols: tuple[str, ...],
    basis: str,
    charge: int,
    spin: int,
    cart: bool,
    verbose: int,
    geometry_grad_policy: str,
    include_dipole_integrals: bool,
) -> tuple[Array, Array, Array | None]:
    intor_args = (
        coords_bohr,
        symbols,
        basis,
        int(charge),
        int(spin),
        bool(cart),
        int(verbose),
    )
    overlap = libcint_int1e_with_coords(
        *intor_args,
        "int1e_ovlp",
        None,
        geometry_grad_policy,
    )
    kinetic = libcint_int1e_with_coords(
        *intor_args,
        "int1e_kin",
        None,
        geometry_grad_policy,
    )
    v_nuc = libcint_int1e_with_coords(
        *intor_args,
        "int1e_nuc",
        None,
        geometry_grad_policy,
    )
    dipole_integrals = None
    if include_dipole_integrals:
        dipole_integrals = libcint_int1e_with_coords(
            *intor_args,
            "int1e_r",
            3,
            geometry_grad_policy,
        )
    return overlap, kinetic + v_nuc, dipole_integrals


def _libcint_one_electron_from_mol(
    *,
    _cached_libcint_host_integral: Callable[..., Any],
    mol: Any,
    geometry_anchor: Array,
    geometry_grad_policy: str,
    include_dipole_integrals: bool,
) -> tuple[Array, Array, Array | None]:
    overlap = _cached_libcint_host_integral(
        mol=mol,
        integral_name="int1e_ovlp",
        geometry_anchor=geometry_anchor,
        geometry_grad_policy=geometry_grad_policy,
        loader=lambda: np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_ovlp")),
            dtype=float,
        ),
    )
    hcore = _cached_libcint_host_integral(
        mol=mol,
        integral_name="int1e_kin+int1e_nuc",
        geometry_anchor=geometry_anchor,
        geometry_grad_policy=geometry_grad_policy,
        loader=lambda: np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_kin")),
            dtype=float,
        )
        + np.asarray(
            mol.intor_symmetric(libcint_intor_name(mol, "int1e_nuc")),
            dtype=float,
        ),
    )
    dipole_integrals = None
    if include_dipole_integrals:
        dipole_integrals = _cached_libcint_host_integral(
            mol=mol,
            integral_name="int1e_r",
            geometry_anchor=geometry_anchor,
            geometry_grad_policy=geometry_grad_policy,
            loader=lambda: np.asarray(
                mol.intor_symmetric(libcint_intor_name(mol, "int1e_r"), comp=3),
                dtype=float,
            ),
        )
    return overlap, hcore, dipole_integrals


