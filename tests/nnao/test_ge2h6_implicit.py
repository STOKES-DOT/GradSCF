"""S-DZ Ge2H6 must have a finite implicit contraction response."""
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool


def test_ge2h6_implicit_contraction_gradient(tmp_path):
    row = dict(structure_id='Ge2H6', symbols=['Ge','Ge']+['H']*6,
        charge=0,spin=0,weight=1,coords_angstrom=[
            [0.,0.,-1.2112570131844733], [0.,0.,1.2112570131844733],
            [1.2482851666586214,.7206977769957784,-1.7450728490982197],
            [0.,-1.4413955539915464,-1.7450728490982197],
            [-1.2482851666586214,.7206977769957784,-1.7450728490982197],
            [1.2482851666586214,-.7206977769957784,1.7450728490982197],
            [0.,1.4413955539915464,1.7450728490982197],
            [-1.2482851666586214,-.7206977769957784,1.7450728490982197]])
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                basis_family='nnao_dzp_s',core_primitives='6-31g')
    problem,_=prepare_fixed_pool(row,layout,tmp_path/'integrals.npz',storage='mmap')
    c=np.asarray(layout.reference_outputs())
    c=c/np.maximum(np.linalg.norm(c,axis=-1,keepdims=True),1e-30)
    energy,gradient,info=problem.evaluate(jnp.asarray(c))
    assert info['stable'] and info['converged']
    assert np.isfinite(gradient).all()
    assert energy < -4142.60
    direction=np.cos(np.arange(c.size)).reshape(c.shape)
    direction/=np.linalg.norm(direction)
    step=1e-4
    plus=problem.evaluate(c+step*direction,gradient=False)[0]
    minus=problem.evaluate(c-step*direction,gradient=False)[0]
    np.testing.assert_allclose(np.sum(gradient*direction),(plus-minus)/(2*step),
                               atol=3e-6,rtol=3e-5)
