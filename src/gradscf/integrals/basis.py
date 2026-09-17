"""Canonical Cartesian basis types and construction interfaces."""
from .backends.jax_reference.basis import (
    CartesianAO, CartesianBasis, ContractedShell, PairBatchGroup,
    ShellPairBatchGroup, QuartetBatchGroup, ShellQuartetBatchGroup,
    basis_from_spec, basis_from_pyscf_spec,
    basis_from_molecule_spec, cartesian_angular_tuples,
)

from .normalization import _normalize_raw_shell_coefficients

from dataclasses import dataclass
import jax
from jaxtyping import Array


@dataclass(frozen=True)
class BasisTopology:
    """Static shell structure. Values of trainable parameters are not stored here."""

    angular_momenta: tuple[int, ...]
    primitive_counts: tuple[int, ...]
    contraction_counts: tuple[int, ...]
    nuclear_charges: tuple[int, ...]
    cart: bool = True

    @property
    def nao(self):
        return sum(((l+1)*(l+2)//2 if self.cart else 2*l+1)*nc
                   for l, nc in zip(self.angular_momenta, self.contraction_counts))

    def __post_init__(self):
        n = len(self.angular_momenta)
        if not n or len(self.primitive_counts) != n or len(self.contraction_counts) != n:
            raise ValueError("Shell topology arrays must have the same nonzero length.")
        if any(l < 0 for l in self.angular_momenta):
            raise ValueError("Angular momenta must be nonnegative.")
        if any(n < 1 for n in self.primitive_counts + self.contraction_counts):
            raise ValueError("Shell primitive/contraction counts must be positive.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasisParameters:
    """Dynamic raw basis values; coordinates in Bohr and exponents in Bohr^-2."""

    exponents: tuple[Array, ...]
    coefficients: tuple[Array, ...]
    centers: Array
    nuclear_coords: Array

    def tree_flatten(self):
        return (self.exponents, self.coefficients, self.centers, self.nuclear_coords), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)


__all__ = [
    "BasisTopology", "BasisParameters", "CartesianAO", "CartesianBasis", "ContractedShell",
    "PairBatchGroup", "ShellPairBatchGroup", "QuartetBatchGroup", "ShellQuartetBatchGroup",
    "basis_from_spec", "basis_from_pyscf_spec",
    "basis_from_molecule_spec", "cartesian_angular_tuples",
]


from .basis_data import load_basis_from_snapshot

def prepare_basis(atom, basis, *, unit="Angstrom", charge=0, spin=0, cart=True):
    """Parse a named/raw basis once into static topology and raw JAX parameters.

    Parameters are not sorted or normalized here. Subsequent evaluations use
    the supplied parameter arrays and include their normalization derivatives.
    Nuclear charges are host-static nonnegative int32 values; zero-charge
    centers supplied through ``MoleculeSpec`` are preserved.
    """
    import jax.numpy as jnp
    import numpy as np
    from gradscf.data.molecule import MoleculeSpec, parse_molecule_spec
    from gradscf.integrals.basis import BasisTopology, BasisParameters

    spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(
        atom, unit=unit, charge=charge, spin=spin)
    charge_error = "Basis nuclear charges must be static nonnegative int32 integers, one per atom."
    try:
        charges = np.asarray(spec.charges)
    except (TypeError, ValueError) as exc:
        raise ValueError(charge_error) from exc
    if (charges.shape != (len(spec.symbols),) or charges.dtype.kind not in "iuf"
            or not np.all(np.isfinite(charges)) or np.any(charges < 0)
            or np.any(charges > np.iinfo(np.int32).max)
            or np.any(charges != np.trunc(charges))):
        raise ValueError(charge_error)
    ls, nprim, nctr, exponents, coefficients, centers = [], [], [], [], [], []
    for i, symbol in enumerate(spec.symbols):
        value = basis[symbol] if isinstance(basis, dict) else basis
        blocks = load_basis_from_snapshot(value, symbol) if isinstance(value, str) else value
        for block in blocks:
            l, rows = int(block[0]), block[1:]
            if not rows or not isinstance(rows[0], (list, tuple)):
                raise ValueError("Expected nonrelativistic [l, [exponent, coefficients...], ...] shells.")
            data = jnp.asarray(rows)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Each primitive requires an exponent and contraction coefficients.")
            ls.append(l)
            nprim.append(data.shape[0])
            nctr.append(data.shape[1]-1)
            exponents.append(data[:, 0])
            coefficients.append(data[:, 1:])
            centers.append(jnp.asarray(spec.coords_bohr[i]))
    topology = BasisTopology(tuple(ls), tuple(nprim), tuple(nctr),
                              tuple(int(z) for z in charges), bool(cart))
    parameters = BasisParameters(tuple(exponents), tuple(coefficients),
                                  jnp.stack(centers), jnp.asarray(spec.coords_bohr))
    return topology, parameters
