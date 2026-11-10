"""Complex body gradients pack real/imaginary rotations without Cayley AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.scf import gks, orbital_optimization as orbital


def problem(xc, collinear, clip=None):
    rng = np.random.default_rng(1102)
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    c, _ = np.linalg.qr(raw)
    h = jnp.asarray(.2*(raw+raw.conj().T))
    factors = rng.normal(size=(3, 2, 2))*.1
    factors = (factors+factors.transpose(0, 2, 1))*.5
    ao = jnp.asarray(rng.normal(size=(7, 2))*.4)
    args = dict(hcore=h, eri=jnp.asarray(np.einsum('Qpq,Qrs->pqrs', factors, factors)),
        ao=ao, ao_deriv1=jnp.stack([ao, .11*ao, -.07*ao, .03*ao]),
        grid_weights=jnp.linspace(.02, .1, 7), nuclear_repulsion=jnp.asarray(.2))
    occupations = ((1., 1., 0., 0.),)
    dimension, rotate, density_from = orbital._rotation_functions('gks', occupations)
    config = gks.GKSConfig(xc_spec=xc, collinear=collinear, potential_clip=clip)
    def evaluate(density, parameters, *, compute_potential=True):
        energy, _, fock = gks.generalized_energy_and_fock(density=density[0],
            config=config, compute_potential=compute_potential, **parameters)
        return energy, fock[None] if compute_potential else None
    return occupations, dimension, rotate, density_from, evaluate, jnp.asarray(c)[None], args


@pytest.mark.parametrize('xc,collinear,clip', [('hf', 'col', None),
    ('pbe0', 'col', None), ('lda_x', 'ncol', 0.)])
def test_complex_body_gradient_and_parameter_jvp_match_scalar_chart(xc, collinear, clip):
    occ, dimension, rotate, density_from, evaluate, c, args = problem(xc, collinear, clip)
    def moving(theta):
        return dict(args, hcore=args['hcore']*(1+.03*theta),
            eri=args['eri']*(1+.02*theta), grid_weights=args['grid_weights']*(1+.01*theta))
    def expected(theta):
        return jax.grad(lambda angles: evaluate(density_from(rotate(angles, c)),
            moving(theta), compute_potential=False)[0])(jnp.zeros(dimension))
    def actual(theta):
        return orbital._brillouin_gradient('gks', occ, c, density_from, evaluate, moving(theta))
    point, direction = jnp.asarray(.1), jnp.asarray(1.)
    observed = jax.jit(lambda t: jax.jvp(actual, (t,), (direction,)))(point)
    reference = jax.jit(lambda t: jax.jvp(expected, (t,), (direction,)))(point)
    assert observed[0].shape == (dimension,) and not jnp.iscomplexobj(observed[0])
    for a, b in zip(observed, reference):
        np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
    for step in (2e-4, 1e-4):
        finite = (actual(point+step)-actual(point-step))/(2*step)
        np.testing.assert_allclose(observed[1], finite, atol=3e-7, rtol=3e-7)


def test_complex_body_linearization_matches_chart_at_stationarity():
    occ, dimension, rotate, density_from, evaluate, c, args = problem('hf', 'col')
    args = dict(args, eri=jnp.zeros_like(args['eri']),
        hcore=c[0]@jnp.diag(jnp.array([-2., -1., .5, 1.2]))@c[0].conj().T)
    energy = lambda angles: evaluate(density_from(rotate(angles, c)), args, compute_potential=False)[0]
    body = lambda angles: orbital._brillouin_gradient('gks', occ, rotate(angles, c),
                                                    density_from, evaluate, args)
    origin, vector = jnp.zeros(dimension), jnp.linspace(-.3, .4, dimension)
    np.testing.assert_allclose(jax.grad(energy)(origin), 0., atol=3e-12)
    actual = jax.jvp(body, (origin,), (vector,))
    expected = jax.jvp(jax.grad(energy), (origin,), (vector,))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize('policy', ['checked', 'regularized', 'unchecked'])
def test_gks_body_preserves_regularized_and_unchecked_chart_policy(monkeypatch, policy):
    from gradscf.scf.autodiff import SCFDifferentiationConfig
    occ, _, _, _, _, c, args = problem('hf', 'col')
    args = dict(args, eri=jnp.zeros_like(args['eri']),
        hcore=c[0]@jnp.diag(jnp.array([-2., -1., .5, 1.2]))@c[0].conj().T)
    original, observed = orbital.attach_scf_backward, []
    def attach(*a, **kw):
        observed.append(kw['residual'].__name__)
        return original(*a, **kw)
    monkeypatch.setattr(orbital, 'attach_scf_backward', attach)
    config = SCFDifferentiationConfig(mode='implicit',
        regularization=.2 if policy == 'regularized' else 0.,
        require_converged=policy != 'unchecked')
    state = orbital.minimize_gks_from_integrals(overlap=jnp.eye(2), **args,
        mo_coeff=c[0], mo_occ=occ[0], max_iterations=0,
        gradient_tolerance=1e-9, differentiation=config)
    assert bool(state.stationary)
    assert observed == ['tangent_residual' if policy == 'checked' else 'tangent_energy']
