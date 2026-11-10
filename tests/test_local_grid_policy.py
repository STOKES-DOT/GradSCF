"""One opt-in screening control, shared execution budget, explicit export scope."""
from dataclasses import fields

import jax.numpy as jnp
import pytest

from gradscf import dft, gto
from gradscf.scf import rks, uks, gks
from gradscf.scf.inputs import grid, assembly
from gradscf.integrals.grids.blocks import AOGrid


@pytest.mark.parametrize('config',[rks.RKSConfig,uks.UKSConfig,gks.GKSConfig])
def test_screening_is_explicit_default_off_and_validated(config):
    assert 'grid_screen_tol' in {f.name for f in fields(config)}, 'Missing common grid_screen_tol'
    assert config().grid_screen_tol==0.
    assert config(grid_screen_tol=1e-14).grid_screen_tol==1e-14
    for cutoff in (-1.,float('inf'),float('nan')):
        with pytest.raises(ValueError,match='grid_screen_tol'):
            config(grid_screen_tol=cutoff)


@pytest.mark.parametrize('method',[dft.RKS,dft.UKS,dft.GKS])
def test_facades_forward_one_control_and_guard_unadapted_response_export(method):
    assert 'grid_screen_tol' in {f.name for f in fields(method)}, 'Missing facade control'
    mol=gto.M(atom='H 0 0 0; H 0 0 1.4',unit='Bohr',basis='sto-3g')
    mf=method(mol,xc='pbe',grid_screen_tol=1e-14)
    assert mf._config().grid_screen_tol==1e-14
    message = ('Generalized spinor response' if method is dft.GKS else
               '(?i)screened.*response|response.*screened')
    with pytest.raises(NotImplementedError,match=message):
        mf._ensure_reference()


def test_positive_control_retains_raw_provider_for_local_execution():
    assert 'grid_screen_tol' in {f.name for f in fields(rks.RKSConfig)}, 'Missing common control'
    cfg=rks.RKSConfig(xc_spec='pbe',grid_screen_tol=1e-14,jk_backend='direct',max_memory=512.)
    inputs=assembly.build_rks_integral_inputs(atom='H 0 0 0; H 0 0 1.4',basis='sto-3g',
        unit='Bohr',config=cfg,xc_spec='pbe',grids_level=0,include_dipole_integrals=False)
    assert isinstance(inputs.grid_ao,AOGrid),'Opt-in local execution needs raw current basis parameters'
    prepared=grid.prepare_scf_grid(inputs.grid_ao,max_memory=512.,screen_tol=cfg.grid_screen_tol)
    assert callable(getattr(prepared,'reduce_local',None))
    assert prepared.raw is inputs.grid_ao
    assert prepared.capacity<=inputs.basis.nao


def test_local_policy_never_admits_only_a_compact_budget():
    import inspect
    assert 'screen_tol' in inspect.signature(grid.prepare_scf_grid).parameters,'Missing bounded local planning'
    n=880
    raw=AOGrid(jnp.zeros((10,3)),jnp.ones(10),jnp.zeros((n,3)),jnp.ones((n,6)),
        jnp.ones((n,6)),(6,)*n,((0,0,0),)*n,4096,adaptive=True)
    with pytest.raises(ValueError,match='local|screen|workspace'):
        grid.prepare_scf_grid(raw,max_memory=1.,screen_tol=1e-14)
