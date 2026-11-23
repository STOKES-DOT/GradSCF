"""Open local-potential binding preserves staged AD and shared physical maps."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.model.autoxc import Candidate,Parameter


def inputs():
    grid=AOGrid(jnp.array([[.1*i,.2,.07*i] for i in range(7)]),jnp.linspace(.1,.3,7),
        jnp.array([[0.,0.,0.],[.5,0.,.1]]),jnp.array([[.7],[1.1]]),
        jnp.ones((2,1)),(1,1),((0,0,0),)*2,3)
    target=iks.Target(jnp.diag(jnp.array([2.,0.])),jnp.eye(2),2)
    problem=iks.WuYang(target,jnp.eye(2),jnp.diag(jnp.array([-1.,1.])),baseline=jnp.zeros((2,2)))
    return problem,grid,{'n':jnp.linspace(.1,.7,7),'u':jnp.linspace(.1,.8,7)}


def test_candidate_designed_density_kernel_matches_dense_wu_yang_gradient():
    from gradscf.model.autoxc.bindings import bind_local_potential

    problem, grid, fields = inputs()
    candidate = Candidate(
        'a*(ambient-n)',
        (Parameter('a', .13, -.5, .5), Parameter('width', 4., 1., 12.)),
        definitions=(('shape', '1+n'), (
            'ambient', 'pair_average(n, exp(-width*r2*(shape+source_shape)))',
        )),
    )
    assert candidate.features == {'n'}
    assert Candidate.from_json(candidate.to_dict()) == candidate
    bound = bind_local_potential(problem, grid, candidate, fields, max_memory=50., block_size=3)
    density = fields['n']
    r2 = jnp.sum((grid.coords[:, None] - grid.coords[None, :])**2, axis=-1)

    def dense_potential(vector):
        amplitude, width = vector
        weights = grid.weights[None, :] * jnp.exp(
            -width*r2*(2+density[:, None]+density[None, :])
        )
        values = amplitude*(weights@density/jnp.sum(weights, axis=1)-density)
        return iks.project_potential(grid, values)

    vector = jnp.asarray([.13, 4.])
    expected = jax.value_and_grad(lambda p: -problem.score(dense_potential(p)).reward)(vector)
    actual = bound.loss_and_grad(vector)
    assert not bound.cached_affine
    assert abs(float(actual[1][1])) > 1e-8
    np.testing.assert_allclose(bound.potential(candidate.initial), dense_potential(vector), atol=1e-12)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-11)


@pytest.mark.parametrize('rank', (1, 2))
def test_candidate_angular_moment_is_rotation_invariant(rank):
    from gradscf.model.autoxc.density import pair_quadrature

    _, grid, fields = inputs()
    density = fields['n']
    displacement = grid.coords[None, :] - grid.coords[:, None]
    distance2 = jnp.sum(displacement**2, axis=-1)
    weight = (density*grid.weights)[None, :] * jnp.exp(-distance2)
    tensor = displacement if rank == 1 else (
        displacement[..., :, None]*displacement[..., None, :] - distance2[..., None, None]*jnp.eye(3)/3
    )
    moment = jnp.einsum('ij,ij...->i...', weight, tensor)
    expected = jnp.sum(moment.reshape(grid.ngrids, -1)**2, axis=1)
    rotation = jnp.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    candidate = Candidate(f'pair_moment(n, exp(-r2), {rank})')
    for coords in (grid.coords, grid.coords@rotation + .8):
        actual = candidate.evaluate({}, fields, pair=pair_quadrature(coords, grid.weights, 3, density.dtype))
        np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_fixed_kernel_is_cached_when_outer_parameters_are_nonlinear():
    from gradscf.model.autoxc.bindings import bind_local_potential

    problem, grid, fields = inputs()
    candidate = Candidate('tanh(a*neighbor)', (Parameter('a', .2, -.5, .5),),
                          definitions=(('neighbor', 'pair_average(n, exp(-r2))'),))
    bound = bind_local_potential(problem, grid, candidate, fields, max_memory=50., block_size=3)
    assert not bound.cached_affine
    assert not bound.pair_block
    distance2 = jnp.sum((grid.coords[:, None] - grid.coords[None, :])**2, axis=-1)
    weights = jnp.exp(-distance2) * grid.weights[None, :]
    expected = iks.project_potential(grid, jnp.tanh(.2*(weights@fields['n'])/weights.sum(axis=1)))
    np.testing.assert_allclose(bound.potential(candidate.initial), expected, atol=1e-13)


def test_joint_wu_yang_fit_recovers_potential_amplitude_and_kernel_width():
    from gradscf.model.autoxc import Evaluator
    from gradscf.model.autoxc.bindings import bind_local_potential

    problem, grid, fields = inputs()
    candidate = Candidate(
        'a*(environment-n)',
        (Parameter('a', .25, .05, 1.), Parameter('width', 3., 1., 8.)),
        definitions=(('environment', 'pair_average(n, exp(-width*r2))'),),
    )
    truth = {'a': .45, 'width': 5.}
    r2 = jnp.sum((grid.coords[:, None] - grid.coords[None, :])**2, axis=-1)
    weights = jnp.exp(-truth['width']*r2) * grid.weights[None, :]
    cases = {}
    for name, density in (('one', fields['n']), ('two', fields['n']**3),
                          ('three', fields['n'].at[0].set(.9))):
        values = truth['a']*(weights@density/weights.sum(axis=1)-density)
        base = problem.base_fock - iks.project_potential(grid, values)
        sample = iks.WuYang(problem.target, problem.kinetic, base, baseline=problem.baseline_potential)
        cases[name] = sample, {'n': density}

    def bind(proposal):
        return {name: bind_local_potential(sample, grid, proposal, fields, block_size=3)
                for name, (sample, fields) in cases.items()}

    evaluator = Evaluator(bind, sample_ids=tuple(cases), fingerprint='pair-kernel-test',
                          streaming=True, maxiter=80, gradient_tolerance=1e-7,
                          gradient_only=True, require_convergence=True)
    evaluation = evaluator.evaluate(candidate)
    assert evaluation.valid and evaluation.fit['gradient_converged'], evaluation.fit
    assert evaluation.parameters == pytest.approx(truth, abs=2e-4)


def test_full_q_path_uses_all_source_blocks_for_candidate_feature():
    from gradscf.model.autoxc.workflows.xc100.molecule import _q_path_energy

    problem, grid, _ = inputs()
    density = problem.target.density
    moment = grid.reduce(lambda total, ao, _, lap, coords, weights:
        total + weights @ jnp.sum(ao * (ao @ density), axis=1), jnp.asarray(0.))
    candidate = Candidate('pair_integral(n, 1)')
    np.testing.assert_allclose(_q_path_energy(grid, density, candidate, {}, 1e-12),
                               .5 * moment**2, atol=1e-12)


def test_shared_projection_pullback_is_the_physical_transpose():
    from gradscf.dft.iks.potential import project_potential_pullback
    _,grid,_=inputs();value=jnp.linspace(.02,.08,7);adjoint=jnp.array([[.2,.1],[.1,-.3]])
    actual=project_potential_pullback(grid,adjoint,max_memory=50.)
    expected=jax.grad(lambda v:jnp.sum(iks.project_potential(grid,v,max_memory=50.)*adjoint))(value)
    np.testing.assert_allclose(actual,expected,atol=1e-13)
    def dual(coords):
        from dataclasses import replace
        moved=replace(grid,centers=coords)
        return jnp.vdot(project_potential_pullback(moved,adjoint,max_memory=50.),value)
    def primal(coords):
        from dataclasses import replace
        return jnp.sum(iks.project_potential(replace(grid,centers=coords),value,max_memory=50.)*adjoint)
    np.testing.assert_allclose(jax.grad(dual)(grid.centers),jax.grad(primal)(grid.centers),atol=1e-12)


def test_nonlinear_binding_gradient_and_hvp_match_full_ad():
    from gradscf.model.autoxc.bindings import bind_local_potential
    problem,grid,fields=inputs()
    candidate=Candidate('a*n/(1+exp(b)*u)',parameters=(Parameter('a',.03,-.2,.2),Parameter('b',.2,-1.,1.)))
    bound=bind_local_potential(problem,grid,candidate,fields,max_memory=50.,block_size=3)
    vector=jnp.array([.03,.2])
    def objective(v):
        values=candidate.evaluate(dict(zip(('a','b'),v)),fields)
        return -problem.score(iks.project_potential(grid,values,max_memory=50.)).reward
    expected=jax.value_and_grad(objective)(vector);actual=bound.loss_and_grad(vector)
    assert not bound.cached_affine
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-11)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-10)
    direction=jnp.array([.2,-.1])
    hvp=jax.jvp(lambda v:bound.loss_and_grad(v)[1],(vector,),(direction,))[1]
    np.testing.assert_allclose(hvp,jax.jvp(jax.grad(objective),(vector,),(direction,))[1],atol=1e-9)


def test_eight_parameter_affine_binding_keeps_the_cache_fast_path():
    from gradscf.model.autoxc.bindings import bind_local_potential
    problem,grid,fields=inputs();parameters=tuple(Parameter(f'a{i}',0.,-.2,.2) for i in range(8))
    candidate=Candidate('+'.join(f'a{i}*n*u**{i}' for i in range(8)),parameters=parameters)
    bound=bind_local_potential(problem,grid,candidate,fields,max_memory=50.,block_size=3)
    assert bound.cached_affine and bound.panels.shape==(8,2,2)
    vector=jnp.linspace(.001,.008,8)
    expected=problem.score(iks.project_potential(grid,candidate.evaluate(dict(zip((p.name for p in parameters),vector)),fields)))
    np.testing.assert_allclose(bound(dict(zip((p.name for p in parameters),vector))).reward,expected.reward,atol=1e-10)


def test_open_binding_obeys_the_projection_memory_budget():
    from gradscf.model.autoxc.bindings import bind_local_potential
    problem,grid,fields=inputs()
    with pytest.raises(ValueError):bind_local_potential(problem,grid,Candidate('n'),fields,max_memory=1e-8)


def test_affine_cache_falls_back_when_cartesian_panels_exceed_budget():
    from gradscf.model.autoxc.bindings import bind_local_potential
    problem,_,fields=inputs();n=100
    grid=AOGrid(jnp.array([[.1*i,.2,.07*i] for i in range(7)]),jnp.linspace(.1,.3,7),
        jnp.zeros((n,3)),jnp.ones((n,1)),jnp.ones((n,1)),(1,)*n,((0,0,0),)*n,3)
    transform=jnp.zeros((n,2)).at[0,0].set(1.).at[1,1].set(1.)
    parameters=tuple(Parameter(f'a{i}',0.,-.2,.2) for i in range(8))
    candidate=Candidate('+'.join(f'a{i}*n*u**{i}' for i in range(8)),parameters=parameters)
    expected=bind_local_potential(problem,grid,candidate,fields,transform=transform,
                                   max_memory=1.,cache_affine=False)
    actual=bind_local_potential(problem,grid,candidate,fields,transform=transform,max_memory=1.)
    assert not actual.cached_affine and not expected.cached_affine
    np.testing.assert_allclose(actual.potential(candidate.initial),expected.potential(candidate.initial))


def test_pullback_respects_geometry_dtype_and_zero_weight_derivatives():
    from dataclasses import replace
    from gradscf.dft.iks.potential import project_potential_pullback
    _,grid,_=inputs()
    grid=replace(grid,exponents=grid.exponents.astype(jnp.float32),
                 coefficients=grid.coefficients.astype(jnp.float32))
    values=jnp.linspace(.02,.08,7,dtype=jnp.float64)
    cotangent=jnp.array([[.2,.1],[.1,-.3]],dtype=jnp.float32)
    actual=project_potential_pullback(grid,cotangent,max_memory=50.)
    expected=jax.grad(lambda v:jnp.sum(iks.project_potential(grid,v)*cotangent))(values)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    grid=replace(grid,weights=grid.weights.at[0].set(0.))
    def primal(w):return jnp.sum(iks.project_potential(replace(grid,weights=w),values)*cotangent)
    def dual(w):return jnp.vdot(project_potential_pullback(replace(grid,weights=w),cotangent),values)
    np.testing.assert_allclose(jax.grad(dual)(grid.weights),jax.grad(primal)(grid.weights),atol=1e-12)


def test_affine_cache_plan_uses_the_actual_formula_precision():
    from gradscf.model.autoxc.bindings import bind_local_potential
    n=100
    grid=AOGrid(jnp.zeros((7,3),dtype=jnp.float32),jnp.ones(7,dtype=jnp.float32),
        jnp.zeros((n,3),dtype=jnp.float32),jnp.ones((n,1),dtype=jnp.float32),
        jnp.ones((n,1),dtype=jnp.float32),(1,)*n,((0,0,0),)*n,3)
    target=iks.Target(jnp.diag(jnp.array([2.,0.],dtype=jnp.float32)),jnp.eye(2,dtype=jnp.float32),2)
    problem=iks.WuYang(target,jnp.eye(2,dtype=jnp.float32),
        jnp.diag(jnp.array([-1.,1.],dtype=jnp.float32)),baseline=jnp.zeros((2,2),dtype=jnp.float32))
    fields={'n':jnp.linspace(.1,.7,7,dtype=jnp.float64),'u':jnp.linspace(.1,.8,7,dtype=jnp.float64)}
    transform=jnp.zeros((n,2),dtype=jnp.float32).at[0,0].set(1.).at[1,1].set(1.)
    candidate=Candidate('+'.join(f'a{i}*n*u**{i}' for i in range(8)),
                        parameters=tuple(Parameter(f'a{i}',0.,-.2,.2) for i in range(8)))
    actual=bind_local_potential(problem,grid,candidate,fields,transform=transform,max_memory=4.)
    assert not actual.cached_affine


def test_external_neural_local_potential_keeps_pytree_parameter_ad():
    problem,grid,_=inputs()
    reporter=iks.WuYang(problem.target,problem.kinetic,problem.base_fock,
                         baseline=problem.baseline_potential,grid=grid)
    parameters={'w':jnp.ones((2,3))*.1,'b':jnp.array([.01,.02,.03]),
                'v':jnp.array([.02,-.01,.03])}
    def model(params,coordinates):
        return jnp.tanh(coordinates[:,:2]@params['w']+params['b'])@params['v']
    objective=lambda p:reporter.local(model,p).reward
    gradient=jax.jit(jax.grad(objective))(parameters)
    direction=jax.tree.map(lambda x:jnp.ones_like(x)*.01,parameters)
    predicted=sum(jnp.vdot(g,d) for g,d in zip(jax.tree.leaves(gradient),jax.tree.leaves(direction)))
    step=1e-4
    plus=jax.tree.map(lambda p,d:p+step*d,parameters,direction)
    minus=jax.tree.map(lambda p,d:p-step*d,parameters,direction)
    np.testing.assert_allclose(predicted,(objective(plus)-objective(minus))/(2*step),atol=1e-7,rtol=1e-5)
