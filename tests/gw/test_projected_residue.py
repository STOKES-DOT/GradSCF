"""Exact residue products with nonsymmetric vertices, active poles and AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw import self_energy
from gradscf.gw.polarizability import rho_response_real


def _inputs(complex_input=False):
    rng = np.random.default_rng(18)
    def value(shape):
        a = rng.normal(size=shape)*.08
        if complex_input:
            a = a+1j*rng.normal(size=shape)*.03
        return jnp.asarray(a)
    energy = jnp.array([-1.1, -.65, .2, .55, .9])
    # Deliberately independent left/right vertices; nocc != nvir.
    return energy, value((7, 5)), value((7, 5)), value((7, 2, 3))


def _dense(omega, energy, left, right, ov, conjugate, static=None):
    ef = -.1
    active = jnp.where(omega > ef, (energy > ef) & (energy < omega),
                       (energy < ef) & (energy > omega))
    sign = jnp.where(omega > ef, 1., -1.)
    if static is None:
        weights = active.astype(jnp.complex128)
    else:
        weights = .5*(jnp.sign(omega-energy)+jnp.sign(energy-ef))
        active = weights != 0
    values = []
    for m in range(energy.size):
        w = jnp.where(active[m], jnp.abs(energy[m]-omega), 0.)
        pi = rho_response_real(w, energy, ov, eta=.02, conjugate=conjugate)
        screened = jnp.linalg.solve(jnp.eye(7)-pi, pi)
        first = left[:, m].conj() if conjugate else left[:, m]
        values.append(jnp.where(active[m], first@screened@right[:, m], 0.))
    values = jnp.stack(values)
    if static is not None:
        values -= static
    result = jnp.sum(weights*values)
    return result if static is not None else sign*result


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("omega", [-.88, .73])
@pytest.mark.parametrize("subtract_static", [False, True])
def test_projected_residue_matches_dense_and_directional_ad(conjugate, omega, subtract_static):
    energy, left, right, ov = _inputs(conjugate)
    static = jnp.arange(5)*.001 if subtract_static else None
    def observed(t):
        return self_energy.sigma_residue_part(omega+.03*t, energy+t*.01, left*(1+t), right,
            ((energy+t*.01, ov*(1-.2*t), 2.),), -.1, .02,
            conjugate=conjugate, wmn_static=static)
    def expected(t):
        return _dense(omega+.03*t, energy+t*.01, left*(1+t), right,
                      ov*(1-.2*t), conjugate, static)
    actual, tangent = jax.jit(lambda t: jax.jvp(observed, (t,), (jnp.ones_like(t),)))(0.)
    reference, derivative = jax.jvp(expected, (0.,), (1.,))
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(tangent, derivative, atol=1e-11, rtol=1e-9)
    reverse = jax.jit(jax.grad(lambda t: observed(t).real))(0.)
    np.testing.assert_allclose(reverse, tangent.real, atol=1e-11, rtol=1e-9)
    for step in (1e-4, 3e-5):
        difference = (observed(step)-observed(-step))/(2*step)
        np.testing.assert_allclose(tangent, difference, atol=1e-9, rtol=2e-6)


def test_molecular_residue_solves_only_vector_rhs(monkeypatch):
    energy, left, right, ov = _inputs()
    original = jnp.linalg.solve
    rhs_shapes = []
    def observe(a, b):
        rhs_shapes.append(b.shape)
        return original(a, b)
    monkeypatch.setattr(jnp.linalg, "solve", observe)
    jax.jit(lambda w: self_energy.sigma_residue_part(w, energy, left, right,
                ((energy, ov, 2.),), -.1, .02))(.73).block_until_ready()
    assert rhs_shapes and all(shape == (7,) for shape in rhs_shapes), rhs_shapes


def test_qp_callbacks_skip_inactive_poles_at_runtime(monkeypatch):
    from gradscf.gw.qp import sigma_cd_batch, qp_residual_batch
    energy, left, right, ov = _inputs()
    observed = []
    original = self_energy._response_solve_one
    def counted(omega, *args, **kwargs):
        jax.debug.callback(lambda w: observed.append(float(w)), omega, ordered=True)
        return original(omega, *args, **kwargs)
    monkeypatch.setattr(self_energy, "_response_solve_one", counted)
    shared = dict(mo_energy=energy, channels=((energy, ov, 2.),),
                  ef=-.1, eta=.02, freqs=jnp.array([.3]), wts=jnp.ones(1))
    stacked = dict(wmn_p=jnp.zeros((2, 1, 5)),
                   b_pm=jnp.stack([left, left*.9]), b_mp=jnp.stack([right, right*1.1]))
    frequencies = jnp.array([.73, .84])
    def evaluate(f):
        return (sigma_cd_batch(f, shared, stacked),
                qp_residual_batch(f, jnp.zeros(2), jnp.zeros(2), shared, stacked))
    jax.block_until_ready(jax.jit(evaluate)(frequencies))
    jax.effects_barrier()
    # Both frequencies enclose the two virtual G poles at .2 and .55.
    # Two callbacks times two targets times two active poles, not all G poles.
    assert len(observed) == 8, observed


def test_projected_physical_direction_including_eta_has_hvp():
    energy, left, right, ov = _inputs()
    def loss(t, reference=False):
        e = energy+t*jnp.array([.01, -.02, .03, -.01, .02])
        l, r, b = left*(1+.2*t), right*(1-.3*t), ov*(1+.1*t)
        w, eta = .73+.01*t, .02+.003*t
        if reference:
            values = []
            for m in (2, 3):
                pi = rho_response_real(jnp.abs(e[m]-w), e, b, eta=eta)
                values.append(l[:, m]@jnp.linalg.solve(jnp.eye(7)-pi, pi)@r[:, m])
            value = sum(values)
        else:
            value = self_energy.sigma_residue_part(w, e, l, r, ((e, b, 2.),), -.1, eta)
        return value.real+.3*value.imag
    derivative = jax.jit(jax.grad(loss))(0.)
    curvature = jax.jit(lambda t: jax.jvp(jax.grad(loss), (t,), (jnp.ones_like(t),))[1])(0.)
    expected = lambda t: loss(t, reference=True)
    np.testing.assert_allclose(derivative, jax.grad(expected)(0.), atol=1e-11, rtol=1e-9)
    np.testing.assert_allclose(curvature, jax.grad(jax.grad(expected))(0.), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("omega", [-.65, .55])
def test_static_half_pole_preserves_the_finite_eta_convention(omega):
    energy, left, right, ov = _inputs()
    static = jnp.arange(5)*.001
    observed = self_energy.sigma_residue_part(omega, energy, left, right,
        ((energy, ov, 2.),), -.1, .02, wmn_static=static)
    np.testing.assert_allclose(observed, _dense(omega, energy, left, right, ov, False, static),
                               atol=1e-12, rtol=1e-10)


def test_live_array_jvp_vjp_transpose_and_directional_difference():
    energy, left, right, ov = _inputs()
    inputs = (energy, left, right, ov, jnp.asarray(.02))
    rng = np.random.default_rng(4)
    direction = tuple(jnp.asarray(rng.normal(size=x.shape)*.01) for x in inputs)
    def value(e, l, r, b, eta):
        sigma = self_energy.sigma_residue_part(.73, e, l, r, ((e, b, 2.),), -.1, eta)
        return jnp.stack([sigma.real, sigma.imag])
    primal, tangent = jax.jit(lambda args, tangents: jax.jvp(value, args, tangents))(inputs, direction)
    _, pullback = jax.vjp(value, *inputs)
    cotangent = jnp.array([.7, -.2])
    reverse = pullback(cotangent)
    inner = sum(jnp.vdot(a, b) for a, b in zip(direction, reverse))
    np.testing.assert_allclose(cotangent@tangent, inner, atol=1e-11, rtol=1e-9)
    assert np.isfinite(primal).all()
    for step in (1e-4, 3e-5):
        plus = tuple(x+step*d for x, d in zip(inputs, direction))
        minus = tuple(x-step*d for x, d in zip(inputs, direction))
        finite = (value(*plus)-value(*minus))/(2*step)
        np.testing.assert_allclose(tangent, finite, atol=1e-10, rtol=1e-5)
