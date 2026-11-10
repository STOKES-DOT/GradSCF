"""Direct grid-density fitting shares the local-potential and spectral kernel."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from test_wu_yang import manufactured


@pytest.mark.parametrize('layout', ['rks','uks'])
def test_local_density_fit_recovers_known_ground_state_density(layout):
    target, basis, base = manufactured(layout)
    result = iks.invert(target, basis, base, method='density_match', regularization=0.,
                        tolerance=1e-6, gradient_tolerance=1e-10, maxiter=300)
    assert result.valid and result.reference.method == 'density_match'
    assert result.method == 'density_match' and result.objective_units == 'dimensionless'
    assert result.relative_density_error < 1e-6
    np.testing.assert_allclose(result.density, target.density, atol=1e-5, rtol=0.)


def test_grid_density_objective_gradient_and_hvp_match_finite_differences(monkeypatch):
    from gradscf.dft.iks import local as wu_yang
    original = wu_yang.minimize_eager
    checked = []
    def inspect(value_grad, initial, **kwargs):
        point = initial+jnp.array([.04,-.02])
        direction = jnp.array([.3,-.4])
        step = 1e-5
        value, gradient = value_grad(point)
        numeric = (value_grad(point+step*direction)[0]-value_grad(point-step*direction)[0])/(2*step)
        np.testing.assert_allclose(jnp.dot(gradient,direction),numeric,atol=2e-10,rtol=2e-6)
        hvp = jax.jvp(lambda x:value_grad(x)[1],(point,),(direction,))[1]
        numeric_hvp = (value_grad(point+step*direction)[1]-value_grad(point-step*direction)[1])/(2*step)
        np.testing.assert_allclose(hvp,numeric_hvp,atol=2e-9,rtol=2e-5)
        checked.append(True)
        return original(value_grad,initial,**kwargs)
    monkeypatch.setattr(wu_yang,'minimize_eager',inspect)
    target,basis,base = manufactured()
    result = iks.invert(target,basis,base,method='density_match',regularization=0.,
                        tolerance=1e-6,gradient_tolerance=1e-10)
    assert checked and result.valid
