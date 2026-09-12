"""Canonical Cartesian basis types and construction interfaces."""
from .backends.jax_reference.basis import (
    CartesianAO, CartesianBasis, ContractedShell, PairBatchGroup,
    ShellPairBatchGroup, QuartetBatchGroup, ShellQuartetBatchGroup,
    basis_from_spec, basis_from_pyscf_spec, basis_from_pyscf_mol_cart,
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
    "basis_from_spec", "basis_from_pyscf_spec", "basis_from_pyscf_mol_cart",
    "basis_from_molecule_spec", "cartesian_angular_tuples",
]
