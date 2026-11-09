"""Actual compact XC consumers, not a full AO tensor with zeros attached."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids import blocks
from gradscf.scf import rks, uks, gks


def _raw(t=0.):
    centers=jnp.asarray([[0.,0.,0.],[.2,.1,0.],[-.3,.2,.1],
                         [18.,0.,0.],[-19.,0.,0.],[0.,21.,0.]])
    coords=jnp.asarray([[.15,.21,.1],[-.3,.1,.2],[.2,-.1,.4],[.3,.2,-.2],[0.,.1,.2]])
    return blocks.AOGrid(coords+t*jnp.asarray([.01,-.02,.03]),
        jnp.asarray([.2,.3,-.01,.15,.12])*jnp.exp(.1*t),centers,
        jnp.ones((6,1))*jnp.exp(.03*t),jnp.ones((6,1)),(1,)*6,((0,0,0),)*6,3)


def _evaluate(method,xc,kind,density,grid):
    ao=jnp.zeros((0,6));derivative=jnp.zeros((4,0,6))
    if method=='rks':
        return rks._restricted_xc_energy_and_matrix(density=density,ao=ao,ao_deriv1=derivative,
            weights=grid.weights,cfg=rks.RKSConfig(xc_spec=xc),xc_kind=kind,grid_ao=grid)
    if method=='uks':
        return uks._unrestricted_xc_energy_and_matrices(density_a=.6*density,density_b=.4*density,
            ao=ao,ao_deriv1=derivative,weights=grid.weights,cfg=uks.UKSConfig(xc_spec=xc),
            xc_kind=kind,grid_ao=grid)
    zero=jnp.zeros_like(density)
    spin=jnp.block([[.6*density,zero],[zero,.4*density]]).astype(jnp.complex128)
    cfg=gks.GKSConfig(xc_spec=xc,collinear='ncol' if method=='gks-ncol' else 'col')
    return gks._generalized_xc(spin,ao,derivative,grid.weights,cfg,kind,grid_ao=grid)


@pytest.mark.parametrize('method,xc,kind',[
    ('rks','svwn','LDA'),('rks','pbe','GGA'),('rks','pbe0','GGA'),
    ('rks','cam-b3lyp','GGA'),('rks','r2scan','MGGA'),
    ('uks','pbe','GGA'),('uks','r2scan','MGGA'),('gks','pbe','GGA'),('gks-ncol','svwn','LDA')])
def test_local_consumer_reduces_actual_ao_width_and_preserves_values(method,xc,kind):
    assert hasattr(blocks,'_LocalAOGrid'),'Missing local grid provider'
    raw=_raw();local=blocks._LocalAOGrid.from_grid(raw,cutoff=1e-14,capacity=4)
    widths=[]
    class Observed:
        weights=raw.weights
        def reduce(self,*a,**kw):return local.reduce(*a,**kw)
        def reduce_local(self,function,initial,**kw):
            def record(carry,ao,*rest):
                jax.debug.callback(lambda width:widths.append(int(width)),ao.shape[1],ordered=True)
                return function(carry,ao,*rest)
            return local.reduce_local(record,initial,**kw)
    d=jnp.eye(6)*.3+jnp.ones((6,6))*.01
    expected=_evaluate(method,xc,kind,d,raw)
    actual=jax.jit(lambda x:_evaluate(method,xc,kind,x,Observed()))(d)
    jax.block_until_ready(actual);jax.effects_barrier()
    assert widths and set(widths)=={4},f'XC used full AO work: observed widths={widths}'
    for got,want in zip(actual,expected):
        assert np.isfinite(np.asarray(got)).all()
        np.testing.assert_allclose(got,want,atol=1e-10,rtol=1e-10,equal_nan=False)


@pytest.mark.parametrize('method',['rks','uks','gks'])
def test_local_xc_keeps_density_and_basis_jvp_vjp(method):
    assert hasattr(blocks,'_LocalAOGrid'),'Missing local grid provider'
    d=jnp.eye(6)*.3+jnp.ones((6,6))*.01
    dd=jnp.eye(6)*.02+jnp.ones((6,6))*.003
    def energy(t,dm,local):
        raw=_raw(t)
        grid=blocks._LocalAOGrid.from_grid(raw,cutoff=1e-14,capacity=4) if local else raw
        return _evaluate(method,'pbe','GGA',dm,grid)[0]
    ref=lambda t,dm:energy(t,dm,False)
    opt=lambda t,dm:energy(t,dm,True)
    for f in (ref,opt):
        value,tangent=jax.jvp(f,(jnp.asarray(0.),d),(jnp.asarray(.07),dd))
        grads=jax.grad(f,argnums=(0,1))(jnp.asarray(0.),d)
        for array in (value,tangent,*grads):assert np.isfinite(np.asarray(array)).all()
    np.testing.assert_allclose(jax.jvp(opt,(jnp.asarray(0.),d),(jnp.asarray(.07),dd)),
        jax.jvp(ref,(jnp.asarray(0.),d),(jnp.asarray(.07),dd)),atol=1e-9,rtol=1e-9)
    for a,b in zip(jax.grad(opt,argnums=(0,1))(0.,d),jax.grad(ref,argnums=(0,1))(0.,d)):
        np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9,equal_nan=False)
