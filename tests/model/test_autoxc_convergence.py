"""A fit is certified by projected gradient, not objective-change or budget."""
import jax
import jax.numpy as jnp
import numpy as np


def test_bounded_gradient_certificate_allows_a_nonzero_raw_gradient():
    from gradscf.solvers.nonlinear import minimize_eager
    result=minimize_eager(jax.value_and_grad(lambda x:jnp.sum((x-2.)**2)),
        jnp.zeros(1),bounds=[(-.5,.5)],gradient_only=True,gradient_tolerance=1e-9)
    assert result.success and result.projected_gradient_max<=1e-9
    assert np.max(np.abs(result.jac))>1.


def test_gradient_only_does_not_accept_a_sub_ulp_energy_change():
    from gradscf.solvers.nonlinear import minimize_eager
    def flat(x):return jnp.asarray(1e12),jnp.ones_like(x)*.01
    result=minimize_eager(flat,jnp.zeros(1),gradient_only=True,gradient_tolerance=1e-6,maxiter=5)
    assert not result.success and result.projected_gradient_max>1e-6


def test_unconverged_inner_fit_cannot_be_an_accepted_candidate():
    from gradscf.model.autoxc import Candidate,Evaluator,Parameter
    from gradscf.dft.iks import WuYangScore
    def score(p):
        value=-(p['a']-.137)**4
        return WuYangScore(0.,0.,value,0.,jnp.eye(2),jnp.eye(2),1.,0.,jnp.nan,True)
    evaluator=Evaluator(lambda c:{'one':score},sample_ids=('one',),fingerprint='strict-fit',
        maxiter=1,gradient_tolerance=1e-14,gradient_only=True,require_convergence=True)
    result=evaluator.evaluate(Candidate('a*n',parameters=(Parameter('a',0.,-.2,.2),)))
    assert not result.valid and result.reward is None and not result.fit['gradient_converged']


def test_inner_wall_time_budget_is_a_failure_guard(monkeypatch):
    import time
    from gradscf.model.autoxc import Candidate,Evaluator,Parameter
    from gradscf.dft.iks import WuYangScore
    ticks=iter([0.,2.,3.,4.,5.])
    monkeypatch.setattr(time,'monotonic',lambda:next(ticks,100.))
    def score(p):return WuYangScore(0.,0.,-(p['a']-.1)**2,0.,jnp.eye(2),jnp.eye(2),1.,0.,jnp.nan,True)
    evaluator=Evaluator(lambda c:{'one':score},sample_ids=('one',),fingerprint='time-budget',max_seconds=1.)
    result=evaluator.evaluate(Candidate('a*n',parameters=(Parameter('a',0.,-.2,.2),)))
    assert not result.valid and result.reward is None
    assert 'time' in result.error.lower() and not result.fit['success']


def test_wall_budget_is_checked_after_fitting_even_without_parameters(monkeypatch):
    from types import SimpleNamespace
    from gradscf.model.autoxc import Candidate,Evaluator
    from gradscf.model.autoxc import loop
    from gradscf.dft.iks import WuYangScore
    sample=WuYangScore(0.,0.,0.,0.,jnp.eye(2),jnp.eye(2),1.,0.,jnp.nan,True)
    ticks = iter([0., 2.])
    monkeypatch.setattr(loop, 'time', SimpleNamespace(monotonic=lambda: next(ticks, 2.)))
    result=Evaluator(lambda c:{'one':lambda p:sample},sample_ids=('one',),
        fingerprint='constant-budget',max_seconds=1.).evaluate(Candidate('n'))
    assert not result.valid and result.reward is None and 'time' in result.error.lower()


def test_a_final_converged_step_cannot_overrun_the_fit_budget(monkeypatch):
    import time
    from types import SimpleNamespace
    import gradscf.model.autoxc.loop as module
    from gradscf.model.autoxc import Candidate,Evaluator,Parameter
    from gradscf.dft.iks import WuYangScore
    clock=[0.];monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    def minimizer(fn,initial,**kw):
        fn(initial);clock[0]=2.
        return SimpleNamespace(x=initial,success=True,nit=1,nfev=1,message='converged',
                               jac=jnp.zeros_like(initial),projected_gradient_max=0.)
    monkeypatch.setattr(module,'minimize_eager',minimizer)
    def score(p):return WuYangScore(0.,0.,-(p['a']-.1)**2,0.,jnp.eye(2),jnp.eye(2),1.,0.,jnp.nan,True)
    result=Evaluator(lambda c:{'one':score},sample_ids=('one',),fingerprint='last-step-budget',
        max_seconds=1.).evaluate(Candidate('a*n',parameters=(Parameter('a',.1,-.2,.2),)))
    assert not result.valid and result.reward is None and 'time' in result.error.lower()
