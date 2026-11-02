import numpy as np
import pytest
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.pool import active_pool_basis,pool_projection
from family import lift,active_mask

@pytest.mark.parametrize('tier',['s','m','l'])
@pytest.mark.parametrize('zeta',[1,2,3])
def test_pool_core_and_nested_radial_space(tier,zeta):
    atom=[('H',(0,0,-1.0)),('C',(0,0,0)),('N',(0,0,1.2))]
    sz=prepare_direct_basis(atom,basis_family=f'nnao_szp_{tier}',core_primitives=6)
    lower=lift(sz,sz.reference_outputs(),max(1,zeta-1))
    target=lift(lower,lower.reference_outputs(),zeta)
    assert target.topology.nao==15+9*zeta
    assert active_pool_basis(sz)[0]==active_pool_basis(target)[0]
    assert target.num_channels==3*zeta
    for role,a,b,c,d in zip(sz.roles,sz.parameters.exponents,target.parameters.exponents,
                           sz.parameters.coefficients,target.parameters.coefficients):
        np.testing.assert_array_equal(a,b)
        if role=='core':np.testing.assert_array_equal(c,d)
        assert d.shape[1]==(zeta if role.startswith('valence_') else 1)
    indices=[];offset=0
    for l,old,new in zip(target.topology.angular_momenta,lower.topology.contraction_counts,target.topology.contraction_counts):
        indices.extend(range(offset,offset+old*(2*l+1)));offset+=new*(2*l+1)
    a=np.asarray(pool_projection(lower,lower.reference_outputs()))
    b=np.asarray(pool_projection(target,target.reference_outputs()))
    np.testing.assert_allclose(b[:,indices],a,atol=1e-12)

def test_full_valence_span_is_not_an_optimizable_radial_subspace():
    sz=prepare_direct_basis('H 0 0 0; C 0 0 1',basis_family='nnao_szp_s')
    tz=lift(sz,sz.reference_outputs(),3)
    mask=active_mask(tz)
    for atom,slot,role,n,c in zip(tz.shell_atoms,tz.slots,tz.roles,
                                 tz.topology.primitive_counts,tz.topology.contraction_counts):
        if slot<0:continue
        if role.startswith('valence_'):assert not mask[atom,slot:slot+c,:n].any()
        else:assert mask[atom,slot:slot+c,:n].all()
