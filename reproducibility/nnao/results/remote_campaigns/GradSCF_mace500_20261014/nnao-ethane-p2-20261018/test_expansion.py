"""Physical subspace checks for adding only a second carbon valence p shell."""
from pathlib import Path
import json
import jax
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.pool import active_pool_basis,pool_projection
from expansion import expand_carbon_p


def test_carbon_p_expansion_retains_the_original_ao_space_and_primitive_pool():
    row=json.loads((Path(__file__).resolve().parent.parent/'nnao-ethane-torsion-20261018'/'source-geometry.json').read_text())
    lower=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family='nnao_szp_l',core_primitives='6-31g',cart=False)
    coefficients=np.array(lower.reference_outputs())
    expanded=expand_carbon_p(lower,coefficients)
    assert lower.topology.nao==44 and expanded.topology.nao==50
    old_top,old_params=active_pool_basis(lower)
    new_top,new_params=active_pool_basis(expanded)
    assert old_top==new_top and old_top.nao==152
    for name in ['exponents','coefficients','centers','nuclear_coords']:
        for before,after in zip(getattr(old_params,name),getattr(new_params,name)):
            np.testing.assert_array_equal(before,after)
    indices=[];offset=0;added=0
    for atom,role,ell,old,new,c in zip(lower.shell_atoms,lower.roles,lower.topology.angular_momenta,
            lower.topology.contraction_counts,expanded.topology.contraction_counts,expanded.parameters.coefficients):
        target=lower.symbols[atom]=='C' and role=='valence_p'
        assert new==(2 if target else old)
        if target:
            assert np.linalg.matrix_rank(np.asarray(c))==2
            added+=1
        indices.extend(range(offset,offset+old*(2*ell+1)))
        offset+=new*(2*ell+1)
    assert added==2
    before=np.asarray(pool_projection(lower,coefficients))
    after=np.asarray(pool_projection(expanded,expanded.reference_outputs()))
    np.testing.assert_allclose(after[:,indices],before,atol=2e-12,rtol=1e-12)


def test_fixed_core_and_hydrogen_contractions_are_unchanged():
    lower=prepare_direct_basis('C 0 0 0; H 0 0 1',basis_family='nnao_szp_l',core_primitives='6-31g')
    coefficients=lower.reference_outputs()
    expanded=expand_carbon_p(lower,coefficients)
    old_bound=lower.bind(coefficients)
    for owner,role,old,new in zip(lower.shell_atoms,lower.roles,
            old_bound.coefficients,expanded.parameters.coefficients):
        if role=='core' or lower.symbols[owner]=='H':
            np.testing.assert_array_equal(old,new)

