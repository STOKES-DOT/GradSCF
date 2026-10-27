"""Bounded dense orbital-Hessian rescue; no finite-difference checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable


@pytest.mark.parametrize('n,davidson_ok,negative,dense_ok,expected',[
    (24,False,False,True,True),(24,True,False,True,True),
    (24,False,True,True,False),(24,False,False,False,False),
    (66,False,False,True,True),(92,False,False,True,False)])
def test_unresolved_curvature_uses_bounded_dense_rescue(monkeypatch,n,davidson_ok,negative,dense_ok,expected):
    from gradscf.scf import stability as m
    no=n//2;dimension=no*(n-no)
    data=dict(overlap=jnp.eye(n),hcore=jnp.diag(jnp.r_[-jnp.ones(no),jnp.ones(n-no)]),
        df_factors=jnp.zeros((1,n,n)),eri=None,nelectron=2*no,nuclear_repulsion=0.,
        ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0),
        config=RKSConfig(xc_spec='hf',jk_backend='df'))
    result=run_rks_from_integrals_traceable(**data)
    diagonal=jnp.arange(dimension,dtype=jnp.float64)+1.
    if negative:diagonal=diagonal.at[0].set(-1.)
    monkeypatch.setattr(m,'_df_rhf_orbital_curvature',
        lambda *args:(jnp.zeros(dimension),lambda v:diagonal*v))
    original=m.solve_hermitian;events=[]
    def solve(operator,*,config):
        jax.debug.callback(lambda dense:events.append(bool(dense)),config.method=='dense',ordered=True)
        if config.method=='dense':
            solved=original(operator,config=config)
            return solved if dense_ok else solved._replace(converged=jnp.zeros_like(solved.converged))
        # Isolate the retry protocol, independent of Davidson's convergence on
        # this synthetic operator. The dense branch uses its actual HVP.
        solved=original(jnp.diag(jnp.arange(4,dtype=jnp.float64)+1),
            config=m.EigenSolverConfig(method='dense',nroots=3,atol=config.atol))
        return solved._replace(vectors=jnp.eye(dimension,3),raw_vectors=jnp.eye(dimension,4),
            converged=jnp.full((3,),davidson_ok),residual_norms=jnp.full((3,),0. if davidson_ok else 1e-6))
    monkeypatch.setattr(m,'solve_hermitian',solve)
    _,info=jax.jit(lambda:m.stabilize_df_rhf(result,**data,max_restarts=0))()
    jax.block_until_ready(info)
    dense_attempt=not davidson_ok and dimension<=2048
    assert bool(info['stable'])==expected
    assert bool(info.get('stability_dense_fallback',False))==dense_attempt
    assert events==([False] if davidson_ok else [False,False]+([True] if dense_attempt else []))
    if dense_attempt and dense_ok:
        np.testing.assert_allclose(info['minimum_curvature'],-1. if negative else 1.,atol=1e-12)
        assert float(info['stability_eigen_residual'])<=1e-7
