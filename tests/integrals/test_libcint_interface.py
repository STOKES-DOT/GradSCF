"""One production Gaussian engine, preserving public integral products."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_libcint_is_the_only_default_integral_interface():
    from gradscf import integrals
    topology,_=integrals.prepare_basis('H 0 0 0; H 0 0 .74','sto-3g')
    assert integrals.make_plan(topology).backend=='libcint'
    caps=integrals.backend_capabilities('libcint')
    assert caps.name=='libcint' and caps.supports('overlap',variable='centers',derivative_order=1)
    assert importlib.util.find_spec('gradscf.integrals.backends.jax_reference') is None


@pytest.mark.parametrize('backend',['cpu','native','jax','jax_reference','pyscf'])
def test_removed_backend_names_have_an_explicit_migration_error(backend):
    from gradscf import integrals
    topology,_=integrals.prepare_basis('He 0 0 0','sto-3g')
    with pytest.raises(ValueError,match='libcint'):
        integrals.make_plan(topology,backend=backend)


def test_public_molecular_products_preserve_libcint_values():
    from gradscf import integrals
    atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
    topology,parameters=integrals.prepare_basis(atom,'sto-3g')
    plan=integrals.make_plan(topology)
    basis=integrals.basis_from_spec(atom,basis='sto-3g')
    np.testing.assert_allclose(integrals.overlap_matrix(basis),plan.evaluate('overlap',parameters),atol=2e-12)
    np.testing.assert_allclose(integrals.build_hcore(basis),plan.evaluate('kinetic',parameters)+plan.evaluate('nuclear',parameters),atol=2e-11)
    np.testing.assert_allclose(integrals.dipole_matrix(basis,origin=jnp.zeros(3)),plan.evaluate('dipole',parameters,origin=jnp.zeros(3)),atol=2e-12)
    np.testing.assert_allclose(integrals.eri_pair_matrix_packed(basis),plan.evaluate('eri',parameters,aosym='s4'),atol=2e-12)


def test_public_integrals_keep_live_geometry_without_an_alternate_engine():
    from gradscf import integrals
    def value(distance):
        atom=[('H',(0.,0.,0.)),('H',(0.,0.,distance))]
        basis=integrals.basis_from_spec(atom,basis='sto-3g',unit='Bohr')
        return jnp.sum(integrals.eri_tensor(basis)**2)
    point=jnp.asarray(1.4)
    analytic=jax.jit(jax.grad(value))(point)
    for step in (2e-4,1e-4):
        expected=(value(point+step)-value(point-step))/(2*step)
        np.testing.assert_allclose(analytic,expected,atol=2e-8,rtol=2e-7)


def test_production_consumers_do_not_import_removed_gaussian_engine():
    import ast
    root=Path(__file__).parents[2]/'src'
    found=[]
    for path in root.rglob('*.py'):
        if 'vendor' in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module and 'jax_reference' in node.module:
                found.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not found,found
