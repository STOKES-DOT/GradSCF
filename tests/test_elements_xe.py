import numpy as np
import pytest
from gradscf.data.molecule import atomic_number, parse_molecule_spec
from gradscf.integrals.grids import BRAGG_RADII,TREUTLER_XI,build_molecular_grid,_build_molecular_grid_from_spec_jax


def test_elements_through_xe():
    for z,s in enumerate('Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe'.split(),37):
        assert atomic_number(s.lower())==z
        assert parse_molecule_spec(f'{s} 0 0 0',charge=1).nelectron==z-1
        assert z in BRAGG_RADII and z in TREUTLER_XI
    with pytest.raises(ValueError): atomic_number('Xx')


@pytest.mark.parametrize('symbol',['Rb','Sr','In','I','Xe'])
def test_heavy_grid_matches_reference(symbol):
    pytest.importorskip('pyscf')
    from pyscf import gto
    from pyscf.dft import gen_grid,radi
    z=atomic_number(symbol)
    mol=gto.M(atom=f'{symbol} 0 0 0; H 0 0 2',basis='3-21g',spin=(z+1)%2,cart=True,verbose=0)
    grids=gen_grid.gen_atomic_grids(mol,level=0,radi_method=radi.treutler_ahlrichs,prune=gen_grid.nwchem_prune)
    cr,wr=gen_grid.get_partition(mol,grids,radii_adjust=radi.treutler_atomic_radii_adjust,atomic_radii=radi.BRAGG_RADII,becke_scheme=gen_grid.original_becke)
    c,w,_=build_molecular_grid(f'{symbol} 0 0 0; H 0 0 2',level=0)
    np.testing.assert_allclose(c,cr,atol=3e-12,rtol=1e-13)
    np.testing.assert_allclose(w,wr,atol=3e-11,rtol=1e-11)


def test_extended_grid_jax_matches_numpy_and_has_coordinate_gradients():
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    spec=parse_molecule_spec('I 0 0 0; H 0 0 1.6')
    expected_c,expected_w,_=build_molecular_grid('I 0 0 0; H 0 0 1.6',level=0)
    coords=jnp.asarray(spec.coords_bohr)
    def evaluate(r):
        c,w=_build_molecular_grid_from_spec_jax(replace(spec,coords_bohr=r),level=0)
        return c,w
    c,w=evaluate(coords)
    np.testing.assert_allclose(c,expected_c,atol=1e-12)
    np.testing.assert_allclose(w,expected_w,atol=1e-10)
    f=lambda r:jnp.sum(evaluate(r)[1]*jnp.exp(-jnp.sum(evaluate(r)[0]**2,axis=1)))
    gradient=jax.grad(f)(coords)
    delta=jnp.zeros_like(coords).at[1,2].set(1e-5)
    np.testing.assert_allclose(gradient[1,2],(f(coords+delta)-f(coords-delta))/2e-5,atol=2e-7,rtol=1e-5)


@pytest.mark.parametrize('symbol,bond',[('F',.92),('Cl',1.27),('Br',1.41),('I',1.61)])
@pytest.mark.parametrize('xc',['hf','pbe'])
def test_halogen_hydride_scf_agrees(symbol,bond,xc):
    pytest.importorskip('pyscf')
    if xc!='hf':pytest.importorskip('jax_xc')
    from pyscf import gto as pgto,scf as pscf,dft as pdft
    from gradscf import gto,scf
    atom=f'H 0 0 0; {symbol} 0 0 {bond}'
    mol=gto.M(atom=atom,basis='3-21g',cart=True)
    ours=scf.RKS(mol,xc=xc,integral_backend='native',max_cycle=120,grids_level=1).run()
    refmol=pgto.M(atom=atom,basis='3-21g',cart=True,verbose=0)
    reference=pscf.RHF(refmol) if xc=='hf' else pdft.RKS(refmol)
    if xc!='hf':reference.xc=xc;reference.grids.level=1
    reference.conv_tol=1e-10;reference.max_cycle=120;reference.init_guess='1e';reference.kernel()
    assert ours.converged and reference.converged
    np.testing.assert_allclose(ours.e_tot,reference.e_tot,atol=2e-7 if xc=='pbe' else 1e-8,rtol=0)
