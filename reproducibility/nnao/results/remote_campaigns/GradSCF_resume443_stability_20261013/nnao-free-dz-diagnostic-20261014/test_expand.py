import numpy as np
import pytest
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.pool import active_pool_basis, pool_projection
from expand import expand_valence_dz

def test_dz_contains_sz_and_preserves_primitive_pool():
    sz=prepare_direct_basis([('H',(0,0,-1.0)),('C',(0,0,0)),('N',(0,0,1.2))],
        basis_family='szp663_direct',core_primitives=6)
    dz=expand_valence_dz(sz,sz.reference_outputs())
    assert sz.topology.nao==24 and dz.topology.nao==33
    assert dz.num_channels==6
    old=sz.bind(sz.reference_outputs());new=dz.bind(dz.reference_outputs())
    for expected,actual in zip(dz.parameters.coefficients,new.coefficients):
        np.testing.assert_allclose(expected,actual,atol=1e-14)
    old_top,old_pool=active_pool_basis(sz);new_top,new_pool=active_pool_basis(dz)
    assert old_top==new_top
    for a,b,c,d in zip(old.exponents,new.exponents,old_pool.coefficients,new_pool.coefficients):
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(c,d)
    for role,a,b in zip(sz.roles,old.coefficients,new.coefficients):
        np.testing.assert_allclose(a[:,0],b[:,0],atol=1e-14)
        assert b.shape[1]==(2 if role.startswith('valence_') else 1)
        if role=='core':np.testing.assert_array_equal(a,b)
    # Original SZ functions are explicit columns of the enlarged DZ projection.
    selected=[];offset=0
    for l,nc in zip(dz.topology.angular_momenta,dz.topology.contraction_counts):
        selected.extend(range(offset,offset+2*l+1));offset+=nc*(2*l+1)
    t0=np.asarray(pool_projection(sz,sz.reference_outputs()))
    t1=np.asarray(pool_projection(dz,dz.reference_outputs()))
    np.testing.assert_allclose(t1[:,selected],t0,atol=1e-13)
    for role,l,a,c in zip(dz.roles,dz.topology.angular_momenta,dz.parameters.exponents,dz.parameters.coefficients):
        if role.startswith('valence_'):
            a=np.asarray(a);c=np.asarray(c)
            metric=(2*np.sqrt(a[:,None]*a[None,:])/(a[:,None]+a[None,:]))**(l+1.5)
            assert abs(c[:,0]@metric@c[:,1])<1e-12

def test_expansion_rejects_a_second_expansion():
    sz=prepare_direct_basis([('H',(0,0,0)),('H',(0,0,.74))],basis_family='szp663_direct')
    dz=expand_valence_dz(sz,sz.reference_outputs())
    with pytest.raises(ValueError,match='single'):
        expand_valence_dz(dz,dz.reference_outputs())
