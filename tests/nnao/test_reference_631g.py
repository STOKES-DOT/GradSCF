"""Single-zeta reference cores: 6-primitive s/p and 3-primitive Ga-Kr d."""
import json
from importlib.resources import files

import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis, supported_elements
from gradscf.model.nnao.initialization import reference_core_blocks, FAMILY_GRID


@pytest.mark.parametrize('symbol', ['Ga','Ge','As','Se','Br','Kr'])
def test_single_d_core_uses_original_three_primitive_contraction(symbol):
    layout=prepare_direct_basis(f'{symbol} 0 0 0', basis_family='nnao_szp_s',core_primitives='6-31g')
    core=[(l,n) for l,n,role in zip(layout.topology.angular_momenta,
        layout.topology.primitive_counts,layout.roles) if role=='core']
    assert core == [(0,6)]*3+[(1,6)]*2+[(2,3)]
    assert layout.topology.nao == 23
    bound=layout.bind(layout.reference_outputs()+.02)
    for role,old,new in zip(layout.roles,layout.parameters.coefficients,bound.coefficients):
        if role=='core':np.testing.assert_array_equal(old,new)


@pytest.mark.parametrize('family',tuple(FAMILY_GRID))
def test_631g_core_does_not_change_active_pool_budget(family):
    layout=prepare_direct_basis('Br 0 0 0',basis_family=family)
    zeta,tier=FAMILY_GRID[family]
    assert layout.topology.nao == 19+4*zeta
    active={role:n for role,n in zip(layout.roles,layout.topology.primitive_counts) if role!='core'}
    assert active==dict(valence_s={'s':3,'m':4,'l':6}[tier],
                        valence_p={'s':3,'m':4,'l':6}[tier],polarization={'s':2,'m':2,'l':3}[tier])


def test_631g_original_records_and_scope():
    from gradscf.data.molecule import atomic_number
    source=json.loads(files('gradscf.model.nnao').joinpath('data/bse_631g_v1.json').read_text())
    assert len(supported_elements(basis_family='nnao_szp_l'))==26
    for symbol in supported_elements(basis_family='nnao_szp_l'):
        parent=[]
        for shell in source['elements'][str(atomic_number(symbol))]['electron_shells']:
            for index,c in enumerate(shell['coefficients']):
                ls=shell['angular_momentum'];l=ls[index] if len(ls)>1 else ls[0]
                parent.append([l,*np.column_stack((np.asarray(shell['exponents'],float),np.asarray(c,float))).tolist()])
        for block in reference_core_blocks(symbol):
            assert block in parent
    assert reference_core_blocks('H')==reference_core_blocks('He')==()
    with pytest.raises(ValueError,match='6-31G|template|supported'):
        prepare_direct_basis('I 0 0 0',basis_family='nnao_szp_l')


def test_old_core_candidate_is_rejected_even_when_active_exponents_match():
    from gradscf.model.nnao.initialization import bind_pool_candidate, pool_channel_keys, core_reference_identity
    layout=prepare_direct_basis('Br 0 0 0',basis_family='nnao_szp_s')
    keys=pool_channel_keys([layout])
    channels=[]
    for l,role,slot,a in zip(layout.topology.angular_momenta,layout.roles,layout.slots,layout.parameters.exponents):
        if slot>=0:
            channels.append(dict(element='Br',l=l,role=role,
                reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                candidate_exponents_bohr_minus2=np.asarray(a).tolist()))
    candidate=dict(parameters=dict(keys=keys,values=np.zeros((len(keys),2)).tolist()),channels=channels)
    with pytest.raises(ValueError,match='core reference'):
        bind_pool_candidate(layout,candidate)
    candidate['core_reference']=core_reference_identity(layout)
    assert bind_pool_candidate(layout,candidate).topology.nao==23
    candidate['core_reference']['policy']='original-bse-6-31g-core-v1'
    with pytest.raises(ValueError,match='core reference'):
        bind_pool_candidate(layout,candidate)
    candidate['core_reference']=core_reference_identity(layout)
    candidate['core_reference']['sha256']='wrong'
    with pytest.raises(ValueError,match='core reference'):
        bind_pool_candidate(layout,candidate)


def test_single_core_projection_and_native_df_coefficient_vjp_agree():
    import jax
    import jax.numpy as jnp
    from gradscf import integrals
    from gradscf.model.nnao.integrals.pool import active_pool_basis,pool_projection
    from gradscf.integrals.molecular.density_fitting import project_factors,unpack_factors
    atom='Br 0 0 0; H 0 0 1.414'
    layout=prepare_direct_basis(atom,basis_family='nnao_tzp_m')
    top,p=active_pool_basis(layout)
    plan=integrals.make_plan(top)
    at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
    aux=integrals.make_auxiliary_plan(top,at)
    metric=aux.metric_factor(p,ap)
    s=plan.evaluate('overlap',p)
    h=plan.evaluate('kinetic',p)+plan.evaluate('nuclear',p)
    b=aux.factors(p,ap,metric_factor=metric)
    native=integrals.make_plan(layout.topology)
    native_aux=integrals.make_auxiliary_plan(layout.topology,at)
    def project(c):
        t=pool_projection(layout,c)
        return t.T@s@t,t.T@h@t,project_factors(b,t)
    def direct(c):
        q=layout.bind(c)
        return native.evaluate('overlap',q),native.evaluate('kinetic',q)+native.evaluate('nuclear',q),unpack_factors(native_aux.factors(q,ap,metric_factor=metric),layout.topology.nao)
    c=layout.reference_outputs()
    a,va=jax.vjp(project,c);z,vz=jax.vjp(direct,c)
    for x,y in zip(a,z):np.testing.assert_allclose(x,y,atol=1e-8,rtol=1e-10)
    cot=tuple(jnp.cos(jnp.arange(x.size)).reshape(x.shape) for x in a)
    np.testing.assert_allclose(va(cot)[0],vz(cot)[0],atol=1e-7,rtol=1e-8)


def test_old_core_cache_identity_is_rejected(tmp_path):
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    row=dict(structure_id='H2',symbols=['H','H'],coords_angstrom=[[0,0,0],[0,0,.74]],charge=0,spin=0,weight=1)
    atom=list(zip(row['symbols'],row['coords_angstrom']))
    old=prepare_direct_basis(atom,basis_family='nnao_szp_s',core_primitives=6)
    new=prepare_direct_basis(atom,basis_family='nnao_szp_s',core_primitives='6-31g')
    prepare_fixed_pool(row,old,tmp_path/'cache.npz')
    with pytest.raises(ValueError,match='identity|signature'):
        prepare_fixed_pool(row,new,tmp_path/'cache.npz')
