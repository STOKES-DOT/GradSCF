"""Physical AO ordering and nuclear-weight contracts at the public boundary."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np


def test_public_nuclear_weights_are_not_rounded_and_remain_differentiable():
    from gradscf import integrals
    basis=integrals.basis_from_spec('H 0 0 0',basis={'H':[[0,[1.2,1.]]]},unit='Bohr')
    expected=-2*jnp.sqrt(2*1.2/jnp.pi)
    def potential(weight):
        return integrals.nuclear_attraction_matrix(basis,atom_coords=jnp.zeros((1,3)),
                                                   atom_charges=jnp.asarray([weight]))[0,0]
    np.testing.assert_allclose(potential(.5),.5*expected,atol=2e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(potential))(jnp.asarray(.5)),expected,atol=2e-12)


def test_public_ao_order_follows_shell_indices_for_matrices_and_packed_eri():
    from gradscf import integrals
    from gradscf.integrals.basis import CartesianBasis
    basis=integrals.basis_from_spec('H 0 0 0; H 0 0 1.4',basis='sto-3g',unit='Bohr')
    permuted=CartesianBasis(aos=tuple(reversed(basis.aos)),atom_coords=basis.atom_coords,
        atom_charges=basis.atom_charges,shells=(replace(basis.shells[0],ao_indices=np.array([1])),
                                               replace(basis.shells[1],ao_indices=np.array([0]))))
    dipole=integrals.dipole_matrix(basis,origin=jnp.zeros(3))
    np.testing.assert_allclose(integrals.dipole_matrix(permuted,origin=jnp.zeros(3)),dipole[:,::-1,::-1],atol=2e-12)
    full=integrals.eri_tensor(basis)[::-1,::-1,::-1,::-1]
    i,j=np.tril_indices(2)
    expected=full[i[:,None],j[:,None],i[None,:],j[None,:]]
    np.testing.assert_allclose(integrals.eri_pair_matrix_packed(permuted),expected,atol=2e-12)


def test_public_scalar_eri_retains_basis_and_higher_coordinate_products():
    from gradscf import integrals
    def value(distance):
        basis=integrals.basis_from_spec([('H',(0.,0.,0.)),('H',(0.,0.,distance))],
                                       basis='sto-3g',unit='Bohr')
        return integrals.eri_element(basis,0,0,1,1)
    third=jax.grad(jax.grad(jax.grad(value)))
    actual=jax.jit(third)(jnp.asarray(1.4))
    step=1e-3
    second=jax.jit(jax.grad(jax.grad(value)))
    expected=(second(1.4+step)-second(1.4-step))/(2*step)
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-5)
    fourth=jax.jit(jax.grad(third))(jnp.asarray(1.4))
    third_compiled=jax.jit(third)
    expected_fourth=(third_compiled(1.4+step)-third_compiled(1.4-step))/(2*step)
    np.testing.assert_allclose(fourth,expected_fourth,atol=2e-6,rtol=2e-5)


def test_public_scalar_eri_keeps_raw_coefficient_and_exponent_response():
    from gradscf import integrals
    from gradscf.integrals.basis import CartesianAO,CartesianBasis
    def value(scale,alpha):
        ao=CartesianAO(jnp.zeros(3),(0,0,0),jnp.asarray([alpha]),jnp.asarray([scale]))
        return integrals.eri_element(CartesianBasis((ao,)),0,0,0,0)
    scale,alpha=jnp.asarray(.7),jnp.asarray(1.2)
    actual=jax.jit(jax.grad(value,argnums=(0,1)))(scale,alpha)
    expected=(8*scale**3*jnp.sqrt(alpha/jnp.pi),scale**4/jnp.sqrt(jnp.pi*alpha))
    np.testing.assert_allclose(actual,expected,atol=3e-11,rtol=3e-10)
