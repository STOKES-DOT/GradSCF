from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np


def test_methane_implicit_energy_gradient_matches_reconverged_scf():
    script=Path('tools/optimize_methane_nnao.py')
    assert script.is_file(), 'Methane implicit-SCF experiment is missing'
    module=runpy.run_path(str(script))
    experiment=module['MethaneRHF'](basis_family='szp3')
    x=jnp.zeros((5,2,2))
    energy,grad,info=experiment.evaluate(x)
    assert info['converged'] and info['min_overlap_eigenvalue']>1e-5
    # Carbon s/p and symmetry-shared H s cover every independent shell channel.
    direction=np.zeros((5,2,2));direction[0]=[[.2,-.3],[.4,.1]];direction[1:,0]=[.15,-.2]
    step=1e-4
    e=lambda t:experiment.evaluate(x+t*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(np.sum(np.asarray(grad)*direction),fd,atol=2e-6,rtol=1e-5)
    assert experiment.evaluate(x-1e-3*grad)[0] < energy


def test_methane_implicit_scf_hessian_matches_reconverged_gradient():
    module=runpy.run_path(str(Path('tools/optimize_methane_nnao.py')))
    experiment=module['MethaneRHF'](basis_family='szp3')
    x=jnp.zeros((5,2,2),dtype=jnp.float64)
    direction=jnp.zeros_like(x)
    direction=direction.at[0].set(jnp.array([[.2,-.3],[.4,.1]]))
    direction=direction.at[1:,0].set(jnp.array([.15,-.2]))
    value=lambda outputs:experiment._implicit_value(
        outputs,experiment.ps,experiment.ph,experiment.rep,
    )[0]
    gradient=jax.jit(jax.grad(value))
    implicit_hvp=jax.jit(
        lambda outputs,tangent:jax.jvp(gradient,(outputs,),(tangent,))[1]
    )(x,direction)
    step=1e-4
    finite_difference=(gradient(x+step*direction)-gradient(x-step*direction))/(2*step)
    np.testing.assert_allclose(implicit_hvp,finite_difference,atol=3e-5,rtol=3e-4)
