"""Near-degenerate Kr2 curvature roots must resolve in the fused SCF path."""
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool


def test_kr2_fused_stability_and_implicit_coefficient_gradient(tmp_path):
    row = dict(structure_id='Kr2', symbols=['Kr','Kr'], charge=0, spin=0, weight=1,
               coords_angstrom=[[0.,0.,2.004999857717652], [0.,0.,-2.004999857717652]])
    layout = prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                  basis_family='nnao_szp_s',core_primitives='6-31g')
    problem, _ = prepare_fixed_pool(row,layout,tmp_path/'integrals.npz',storage='mmap')
    coefficients = np.asarray(layout.reference_outputs())
    coefficients = coefficients/np.maximum(np.linalg.norm(coefficients,axis=-1,keepdims=True),1e-30)
    energy, gradient, info = problem.evaluate(jnp.asarray(coefficients))
    assert info['converged'] and info['stable'] and info['stability_resolved']
    assert info['minimum_curvature'] > 0
    assert info['stability_eigen_residual'] <= 1e-7
    assert np.isfinite(energy) and np.isfinite(gradient).all()
    direction = np.cos(np.arange(coefficients.size)).reshape(coefficients.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    plus = problem.evaluate(coefficients+step*direction,gradient=False)[0]
    minus = problem.evaluate(coefficients-step*direction,gradient=False)[0]
    np.testing.assert_allclose(np.sum(gradient*direction),(plus-minus)/(2*step),
                               atol=2e-6,rtol=2e-5)
