"""Prepared physical data must be reusable dynamic inputs to a shared JIT."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dft import iks


def problem(*, shift=0., grid=None):
    target = iks.Target(jnp.diag(jnp.array([2.,0.])),jnp.eye(2),2)
    return iks.WuYang(target,jnp.eye(2),jnp.diag(jnp.array([-1.,1.]))+shift*jnp.eye(2),
                      baseline=jnp.zeros((2,2)),grid=grid)


def test_prepared_scores_do_not_retrace_for_new_matrix_values():
    traces = []
    def score(prepared,potential):
        traces.append(1)
        return prepared.score(potential).value
    compiled = jax.jit(score)
    first, second = problem(), problem(shift=.03)
    v = jnp.array([[.1,.2],[.2,-.1]])
    for prepared,potential in ((first,v),(first,2*v),(second,3*v)):
        np.testing.assert_allclose(compiled(prepared,potential),prepared.score(potential).value,atol=1e-13)
    assert len(traces)==1
    gradient = jax.jit(jax.grad(lambda a: compiled(first,a*v)))(.2)
    current = first.score(.2*v)
    np.testing.assert_allclose(gradient,jnp.sum(v*(current.density-first.target.density)),atol=1e-12)


def test_grid_diagnostic_reuses_the_already_solved_density(monkeypatch):
    plain = problem()
    reporter = problem(grid=(jnp.eye(2),jnp.ones(2)))
    v = jnp.array([[.1,.2],[.2,-.1]])
    expected = reporter.score(v)
    solved = plain.score(v)
    def fail(*args,**kwargs): raise AssertionError('diagnostics performed another diagonalization')
    monkeypatch.setattr('gradscf.dft.iks.wu_yang._aufbau',fail)
    actual = reporter.diagnose(solved)
    np.testing.assert_allclose(actual.density_error,expected.density_error,atol=1e-14)
    np.testing.assert_allclose(actual.value,expected.value,atol=1e-14)
    assert actual.valid
