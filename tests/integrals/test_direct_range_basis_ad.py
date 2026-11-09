"""Static range-separated direct basis products versus full-value FD only."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_direct_parameter_ad import changed, point_and_direction, problem


@pytest.mark.parametrize('field',['coefficients','log_exponents'])
def test_static_range_basis_products_match_two_step_value_fd(problem,field):
    plan,parameters,density=problem
    point,direction=point_and_direction(parameters,field)
    weights=(jnp.sin(density),.3*jnp.cos(density))
    def oracle(value):
        eri=plan.evaluate('eri',changed(parameters,field,value),omega=.3)
        return (jnp.einsum('pqrs,brs->bpq',eri,density),
                jnp.einsum('prqs,brs->bpq',eri,density))
    reference=jax.jit(oracle)
    weighted=lambda out:sum(jnp.sum(x*w) for x,w in zip(out,weights))
    expected=[]
    for step in (2e-4,1e-4):
        push=jax.tree.map(lambda high,low:(high-low)/(2*step),
            reference(point+step*direction),reference(point-step*direction))
        pull=[]
        for i in range(point.size):
            unit=jnp.zeros(point.size).at[i].set(1.).reshape(point.shape)
            pull.append((weighted(reference(point+step*unit))-weighted(reference(point-step*unit)))/(2*step))
        expected.append((push,jnp.stack(pull).reshape(point.shape)))
    direct=lambda p:plan.get_jk(changed(parameters,field,p),density,omega=.3,
        screening_threshold=1e-12,workers=2)
    value,push=jax.jit(lambda p,v:jax.jvp(direct,(p,),(v,)))(point,direction)
    pull=jax.jit(jax.grad(lambda p:weighted(direct(p))))(point)
    for a,b in zip(value,reference(point)):
        np.testing.assert_allclose(a,b,atol=3e-10,rtol=3e-9)
    for target in expected:
        for a,b in zip(jax.tree.leaves((push,pull)),jax.tree.leaves(target)):
            assert np.isfinite(np.asarray(a)).all()
            np.testing.assert_allclose(a,b,atol=3e-7,rtol=4e-6)
    np.testing.assert_allclose(weighted(push),jnp.vdot(pull,direction),atol=3e-9,rtol=3e-9)
