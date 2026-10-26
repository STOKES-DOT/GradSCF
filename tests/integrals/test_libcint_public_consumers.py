"""Current AO leaves, real nuclear weights and small Cartesian component views."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.integrals.basis import CartesianAO, CartesianBasis


def test_updated_aos_are_not_shadowed_by_old_shell_metadata():
    base = integrals.basis_from_spec('H 0 0 0', basis='sto-3g', spin=1)
    def value(scale):
        ao = replace(base.aos[0], coefficients=scale*base.aos[0].coefficients)
        current = CartesianBasis((ao,), shells=base.shells,
            atom_coords=base.atom_coords, atom_charges=base.atom_charges)
        return integrals.overlap_matrix(current)[0, 0]
    np.testing.assert_allclose(value(.7), .49, atol=2e-12)
    np.testing.assert_allclose(jax.grad(value)(.7), 1.4, atol=2e-12)


@pytest.mark.parametrize('container', [lambda x: jnp.array([x]), lambda x: [x], lambda x: (x,)],
                         ids=['array', 'list', 'tuple'])
def test_irrelevant_real_nuclear_weights_do_not_block_basis_integrals(container):
    base = integrals.basis_from_spec('H 0 0 0', basis='sto-3g', spin=1)
    reference = integrals.build_hcore(base)
    kinetic = integrals.kinetic_matrix(base)
    def value(weight):
        current = CartesianBasis(base.aos, shells=base.shells,
            atom_coords=base.atom_coords, atom_charges=container(weight))
        return integrals.build_hcore(current)[0, 0]
    fractional = CartesianBasis(base.aos, shells=base.shells,
        atom_coords=base.atom_coords, atom_charges=jnp.array([.5]))
    np.testing.assert_allclose(integrals.overlap_matrix(fractional), jnp.eye(1), atol=2e-12)
    np.testing.assert_allclose(value(.5), .5*(reference[0, 0]+kinetic[0, 0]), atol=2e-12)
    np.testing.assert_allclose(jax.grad(value)(.5), (reference-kinetic)[0, 0], atol=2e-12)


def test_partial_cartesian_shell_keeps_packed_integrals():
    ao = CartesianAO(jnp.zeros(3), (1, 0, 0), jnp.array([.8]), jnp.array([.7]))
    basis = CartesianBasis((ao,))
    packed = integrals.eri_pair_matrix_packed(basis)
    np.testing.assert_allclose(packed, integrals.eri_tensor(basis).reshape(1, 1), atol=2e-12)


def test_basis_spectral_factorization_uses_default_integral_interface():
    from gradscf.integrals.molecular.factorization import eri_to_df_factors_from_basis
    basis = integrals.basis_from_spec('H 0 0 0; H 0 0 .74', basis='sto-3g')
    factors = eri_to_df_factors_from_basis(basis)
    np.testing.assert_allclose(jnp.einsum('Pij,Pkl->ijkl', factors, factors),
        integrals.eri_tensor(basis), atol=2e-10)
