"""Actual multi-tile s/p/d/f AO cache against independent PySCF derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from pyscf import gto,dft
from pyscf_adapters import basis_from_pyscf_mol_cart
from gradscf.integrals.grids.blocks import AOGrid,_materialize_scf_grid


def test_s_p_d_f_cache_tail_and_center_update_match_pyscf():
    mol=gto.M(atom='H .1 -.2 .3',basis={'H':[[l,[.7-.1*l,1.]] for l in range(4)]},spin=1,cart=True,unit='Bohr',verbose=0)
    basis=basis_from_pyscf_mol_cart(mol,max_l=3)
    points=np.stack([np.linspace(-1.2,1.3,37),np.linspace(.6,-.8,37),np.zeros(37)],axis=1)
    points[4]=mol.atom_coords()[0]
    raw=AOGrid.from_basis(basis,jnp.asarray(points),jnp.full(37,1/37),block_size=8,adaptive=True)
    for shift in (0.,.07):
        moved=replace(raw,centers=raw.centers+shift)
        cache=jax.block_until_ready(_materialize_scf_grid(moved,block_size=8,deriv=2))
        reference=mol.copy().set_geom_(mol.atom_coords()+shift,unit='Bohr')
        ao=dft.numint.eval_ao(reference,points,deriv=2)
        expected=np.stack([*ao[:4],ao[4]+ao[7]+ao[9]],axis=1)
        np.testing.assert_allclose(cache.values,expected,atol=2e-11,rtol=2e-11)
        if shift==0.:first=np.asarray(cache.values)
        else:assert not np.allclose(first,cache.values)
        assert cache.values.shape==(37,5,20)
