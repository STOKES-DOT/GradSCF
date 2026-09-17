from pathlib import Path
import runpy
import jax.numpy as jnp
import numpy as np


def test_methane_stationary_energy_gradient_matches_reconverged_scf():
    script=Path('tools/optimize_methane_nnao.py')
    assert script.is_file(), 'Methane variational experiment is missing'
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
