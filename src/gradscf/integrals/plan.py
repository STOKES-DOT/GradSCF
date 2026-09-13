"""Reusable static integral plans with freshly bound basis parameters."""
from dataclasses import dataclass, replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .basis import (BasisTopology, BasisParameters, CartesianAO, CartesianBasis,
                    ContractedShell, cartesian_angular_tuples)
from .normalization import normalized_shell_coefficients, radial_primitive_norm
from .capabilities import backend_capabilities


def _check_shapes(topology, parameters):
    n = len(topology.angular_momenta)
    if len(parameters.exponents) != n or len(parameters.coefficients) != n:
        raise ValueError("Basis parameter shell counts do not match the topology.")
    if parameters.centers.shape != (n, 3):
        raise ValueError("Basis centers must have shape (nshell,3).")
    if parameters.nuclear_coords.shape != (len(topology.nuclear_charges), 3):
        raise ValueError("Nuclear coordinates do not match the topology.")
    for a, c, np_, nc in zip(parameters.exponents, parameters.coefficients,
                             topology.primitive_counts, topology.contraction_counts):
        if a.shape != (np_,) or c.shape != (np_, nc):
            raise ValueError("Exponent/coefficient shape does not match shell topology.")


def _reference_basis(topology, parameters, *, precompute_eri_groups=True):
    if not topology.cart:
        raise NotImplementedError("The JAX reference backend supports Cartesian basis functions only.")
    aos, shells = [], []
    for l, a, raw, center in zip(topology.angular_momenta, parameters.exponents,
                                 parameters.coefficients, parameters.centers):
        coefficients = normalized_shell_coefficients(l, a, raw) / radial_primitive_norm(l, a)[:, None]
        angulars = tuple(cartesian_angular_tuples(l))
        for column in range(raw.shape[1]):
            start = len(aos)
            coeff = coefficients[:, column]
            aos.extend(CartesianAO(center, angular, a, coeff) for angular in angulars)
            shells.append(ContractedShell(center, angulars, a, coeff,
                                           np.arange(start, len(aos), dtype=np.int32)))
    return CartesianBasis(tuple(aos), shells=tuple(shells),
                          precompute_eri_groups=precompute_eri_groups,
                          atom_coords=parameters.nuclear_coords,
                          atom_charges=jnp.asarray(topology.nuclear_charges))


@dataclass(frozen=True)
class IntegralPlan:
    topology: BasisTopology
    backend: str
    atm: object
    bas: object
    env_size: int

    def pack(self, parameters, *, origin=None):
        """Pack libcint ATM/BAS/ENV without freezing dynamic numerical values.

        Nuclear atoms and zero-charge basis-center atoms are separate. This
        permits floating centers without spuriously moving nuclear potentials.
        """
        _check_shapes(self.topology, parameters)
        if not jax.config.x64_enabled:
            raise ValueError("The native integral backend requires JAX float64 enabled.")
        env = jnp.zeros(self.env_size, dtype=jnp.float64)
        if origin is not None:
            origin = jnp.asarray(origin)
            if origin.shape != (3,):
                raise ValueError("Integral origin must have shape (3,).")
            env = env.at[1:4].set(origin)
        coords = jnp.concatenate([parameters.nuclear_coords, parameters.centers], axis=0)
        for i in range(len(self.atm)):
            offset = int(self.atm[i, 1])
            env = env.at[offset:offset+3].set(coords[i])
        for i, (l, a, c) in enumerate(zip(self.topology.angular_momenta,
                                         parameters.exponents, parameters.coefficients)):
            ae, ce = int(self.bas[i, 5]), int(self.bas[i, 6])
            coeff = normalized_shell_coefficients(l, a, c)
            env = env.at[ae:ae+a.size].set(a)
            env = env.at[ce:ce+c.size].set(coeff.T.ravel())
        return self.atm, self.bas, env

    def evaluate(self, operator, parameters, *, origin=None):
        """Bind current values and evaluate an operator.

        Reference one-electron calls skip ERI layouts. Reference eager ERI
        calls still rebuild their layouts when binding the current parameters.
        """
        caps = backend_capabilities(self.backend)
        if not caps.supports(operator):
            raise NotImplementedError(f"{operator!r} is not supported by {self.backend}.")
        _check_shapes(self.topology, parameters)
        if origin is not None and operator != "dipole":
            raise ValueError("origin is only meaningful for the dipole operator.")
        if origin is not None:
            origin = jnp.asarray(origin)
            if origin.shape != (3,):
                raise ValueError("Integral origin must have shape (3,).")
        if operator == "dipole" and origin is None:
            charges = jnp.asarray(self.topology.nuclear_charges)
            total = sum(self.topology.nuclear_charges)
            origin = (jnp.einsum("a,ar->r", charges, parameters.nuclear_coords) / total
                      if total else jnp.zeros(3))
        if self.backend == "native":
            from .backends.native_geometry import evaluate_geometry
            # Keep basis/environment values separate from geometry, allowing
            # unsupported exponent/coefficient AD to fail explicitly.
            fixed = replace(parameters, centers=jnp.zeros_like(parameters.centers),
                            nuclear_coords=jnp.zeros_like(parameters.nuclear_coords))
            atm, bas, env = self.pack(fixed)
            coords = jnp.concatenate([parameters.nuclear_coords, parameters.centers], axis=0)
            origin = jnp.zeros(3, dtype=env.dtype) if origin is None else origin
            return evaluate_geometry(operator, atm, bas, env, coords, origin,
                                     self.topology.nao, cart=self.topology.cart)
        from .backends import jax_reference
        names = {"overlap": "overlap_matrix", "kinetic": "kinetic_matrix",
                 "nuclear": "nuclear_attraction_matrix", "dipole": "dipole_matrix", "eri": "eri_tensor"}
        kwargs = {"origin": origin} if operator == "dipole" else {}
        basis = _reference_basis(self.topology, parameters, precompute_eri_groups=operator == "eri")
        return getattr(jax_reference, names[operator])(basis, **kwargs)


@lru_cache(maxsize=32)
def make_plan(topology, *, backend="native"):
    """Cache only topology, index/pointer arrays and backend identity."""
    if backend not in {"native", "jax_reference"}:
        raise ValueError("Integral plans support native or jax_reference backends.")
    nnuc = len(topology.nuclear_charges)
    nsh = len(topology.angular_momenta)
    atm = np.zeros((nnuc+nsh, 6), dtype=np.int32)
    atm[:nnuc, 0] = topology.nuclear_charges
    atm[:, 2] = 1  # point nuclear model; basis-center atoms have zero charge
    atm[:, 1] = 20 + np.arange(nnuc+nsh)*3
    offset = 20 + 3*(nnuc+nsh)
    bas = np.zeros((nsh, 8), dtype=np.int32)
    for i, (l, np_, nc) in enumerate(zip(topology.angular_momenta,
                                        topology.primitive_counts, topology.contraction_counts)):
        bas[i, :4] = (nnuc+i, l, np_, nc)
        bas[i, 5] = offset
        bas[i, 6] = offset + np_
        offset += np_ + np_*nc
    atm.setflags(write=False)
    bas.setflags(write=False)
    return IntegralPlan(topology, backend, atm, bas, offset)
