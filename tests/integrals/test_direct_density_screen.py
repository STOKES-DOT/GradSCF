"""Explicit M(D; S): shell-quartet screening reference, not a hidden D mask.

S selects a canonical 8-fold-symmetric ERI subset. Its indicator derivative is
zero almost everywhere; D remains a live linear operand. Deliberately large
cutoff below tests mask algebra, not molecular accuracy or a total Fock bound.
"""
from dataclasses import replace
from types import SimpleNamespace
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis, _selected_jk

CUTOFF = .05
OMEGA = .3


def mask_for(eri, screen, offsets, cutoff=CUTOFF, *, with_j=True, with_k=True):
    """Independent dense oracle for shell L1 bounds and all canonical copies."""
    n = eri.shape[0]
    pairs = [(a, b) for a, b in zip(offsets[:-1], offsets[1:])]
    bound, magnitude = np.zeros((len(pairs), len(pairs))), np.zeros((len(pairs), len(pairs)))
    values = np.abs(np.asarray(screen).reshape(-1, n, n))
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            bound[i, j] = np.sqrt(max(abs(eri[p, q, p, q]) for p in range(a, b) for q in range(c, d)))
            magnitude[i, j] = values[:, a:b, c:d].sum(axis=(1, 2)).max()
    magnitude = np.maximum(magnitude, magnitude.T)
    shells = np.repeat(np.arange(len(pairs)), np.diff(offsets))
    labels = [shells[index] for index in np.indices((n,) * 4)]
    i, j, k, l = labels
    schwarz = bound[i, j] * bound[k, l]
    density_bound = np.zeros_like(eri)
    products = (((i, j), (k, l)) if with_j else ()) + (((i, k), (i, l), (j, k), (j, l)) if with_k else ())
    for a, b in products:
        density_bound = np.maximum(density_bound, magnitude[a, b])
    return (schwarz >= cutoff) & (8 * schwarz * density_bound >= cutoff)


@pytest.fixture(scope='module')
def system():
    top, parameters = integrals.prepare_basis('H 0 0 0; H .2 .1 1.8',
        [[0, [1.1, 1.]], [1, [.7, 1.]]], unit='Bohr', cart=True)
    plan = integrals.make_plan(top)
    offsets = [0]
    for l, count in zip(top.angular_momenta, top.contraction_counts):
        offsets.append(offsets[-1] + count * (l + 1) * (l + 2) // 2)
    eri = {omega: np.asarray(plan.evaluate('eri', parameters, omega=omega)) for omega in (0., OMEGA)}
    n = top.nao
    screen = np.zeros((2, n, n), dtype=complex)
    for batch, shell, value in ((0, 1, .06j), (1, 3, .04*np.exp(.37j))):
        a, b = offsets[shell:shell+2]
        screen[batch, a:b, a:b] = value / (b - a)**2
    screen[0, offsets[0], offsets[2]] = .03j  # intentionally no transpose counterpart
    rng = np.random.default_rng(20261105)
    density = .2 * (rng.normal(size=(2, 3, n, n)) + 1j * rng.normal(size=(2, 3, n, n)))
    return SimpleNamespace(top=top, parameters=parameters, plan=plan, offsets=offsets,
        eri=eri, provider=NativeDirectBasis(plan, parameters), density=jnp.asarray(density), screen=jnp.asarray(screen))


def oracle(system, screen, omega, with_j=True, with_k=True):
    eri = system.eri[omega]
    mask = mask_for(eri, screen, system.offsets, with_j=with_j, with_k=with_k)
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        np.testing.assert_array_equal(mask, mask.transpose(permutation))
    assert mask.any() and not mask.all(), 'Fixture must exercise a partial quartet subset'
    tensor = jnp.asarray(eri * mask)
    def action(d):
        return (jnp.einsum('pqrs,...rs->...pq', tensor, d) if with_j else jnp.zeros_like(d),
                jnp.einsum('prqs,...rs->...pq', tensor, d) if with_k else jnp.zeros_like(d))
    return action


def compare(left, right, tolerance=1e-10):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        assert np.isfinite(np.asarray(a)).all()
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('omega', [0., OMEGA])
@pytest.mark.parametrize('with_j,with_k', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('complex_density', [False, True], ids=['real', 'complex'])
def test_reference_mask_matches_dense_oracle_for_all_selections(system, omega, with_j, with_k, complex_density):
    d = system.density if complex_density else system.density.real
    action = lambda x: _selected_jk(system.provider, x, screen_density=system.screen,
        screening_threshold=CUTOFF, workers=2, omega=omega, with_j=with_j, with_k=with_k)
    compare(jax.jit(action)(d), oracle(system, system.screen, omega, with_j, with_k)(d))


def test_zero_screen_drops_nonzero_payload_without_changing_default(system):
    d, provider = system.density, system.provider
    old = provider.get_jk(d, screening_threshold=CUTOFF, workers=2)
    compare(provider.get_jk(d, screening_threshold=CUTOFF, workers=2, screen_density=None), old, 0.)
    compare(provider.get_jk(d, screening_threshold=CUTOFF, workers=2, screen_density=jnp.zeros((8, 8))),
            (jnp.zeros_like(d), jnp.zeros_like(d)), 0.)
    compare(provider.get_jk(d, screening_threshold=CUTOFF, workers=2, screen_density=jnp.full((1, 8, 8), 100.)), old)
    assert all(np.linalg.norm(np.asarray(x)) > 0 for x in old)


@pytest.mark.parametrize('omega', [0., OMEGA])
@pytest.mark.parametrize('with_j,with_k', [(True,True),(True,False),(False,True)])
def test_fixed_reference_keeps_linearity_adjoint_and_joint_array_ad(system, omega, with_j, with_k):
    d, s = system.density, system.screen
    rng = np.random.default_rng(781)
    direction = jnp.asarray(.03 * (rng.normal(size=d.shape) + 1j * rng.normal(size=d.shape)))
    ds = jnp.asarray(.0003 * (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)))
    expected = oracle(system, s, omega, with_j, with_k)
    action = lambda x, screen: system.provider.get_jk(x, screen_density=screen,
        screening_threshold=CUTOFF, workers=2, omega=omega, with_j=with_j, with_k=with_k)
    value, tangent = jax.jvp(action, (d, s), (direction, ds))
    compare(value, expected(d))
    compare(tangent, expected(direction), 1e-9)  # never screen with the tangent
    compare(action(1.3*d - .4*direction, s), jax.tree.map(lambda a, b: 1.3*a-.4*b, action(d, s), action(direction, s)))
    compare(action(d, jnp.exp(.3j)*s), value)  # complex reference uses absolute magnitude
    mapped = jax.jit(jax.vmap(action))(jnp.stack((d, d+direction)), jnp.stack((s, jnp.zeros_like(s))))
    compare(mapped, jax.tree.map(lambda x: jnp.stack((x, jnp.zeros_like(x))), expected(d)))
    weights = tuple(jnp.asarray(rng.normal(size=x.shape) + 1j*rng.normal(size=x.shape)) for x in value)
    _, pullback = jax.vjp(lambda x: action(x, s), d)
    lhs = sum(jnp.vdot(a, b) for a, b in zip(action(direction, s), weights))
    np.testing.assert_allclose(lhs, jnp.vdot(direction, pullback(weights)[0]), atol=1e-9, rtol=1e-9)
    def loss(function, x, screen):
        return sum(jnp.vdot(v, v).real + .2*jnp.vdot(v, w).real for v, w in zip(function(x, screen), weights))
    dense = lambda x, screen: expected(x)
    gradient = jax.grad(lambda x, screen: loss(action, x, screen), argnums=(0, 1))
    dense_gradient = jax.grad(lambda x, screen: loss(dense, x, screen), argnums=(0, 1))
    compare(jax.jit(gradient)(d, s), dense_gradient(d, s), 1e-9)
    np.testing.assert_array_equal(gradient(d, s)[1], 0.)  # indicator derivative, away from thresholds
    compare(jax.jvp(gradient, (d, s), (direction, ds))[1],
            jax.jvp(dense_gradient, (d, s), (direction, ds))[1], 1e-9)
    for step in (1e-4, 3e-5):
        for sign in (-1, 1):
            np.testing.assert_array_equal(mask_for(system.eri[omega], s + sign*step*ds, system.offsets),
                                          mask_for(system.eri[omega], s, system.offsets))
        fd = jax.tree.map(lambda a, b: (a-b)/(2*step), action(d+step*direction, s+step*ds), action(d-step*direction, s-step*ds))
        compare(tangent, fd, 1e-8)


def test_projected_screen_follows_the_same_transform_as_payload():
    class Capture:
        def get_jk(self, density, *, screen_density=None, **kwargs):
            self.density, self.screen = density, screen_density
            return 2*density, 3*density
    primitive = Capture()
    transform = jnp.arange(24, dtype=jnp.float64).reshape(8, 3) * .01
    provider = ProjectedNativeDirectBasis(primitive, transform)
    d = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3) * .03
    s = jnp.array([[.2, .3j, 0.], [-.1j, .1, .2], [.4, 0., .3]])
    provider.get_jk(d, screen_density=s, screening_threshold=CUTOFF)
    compare(primitive.density, jnp.einsum('pi,...ij,qj->...pq', transform, d, transform), 1e-12)
    compare(primitive.screen, jnp.einsum('pi,ij,qj->pq', transform, s, transform), 1e-12)


def test_zero_cutoff_retains_projected_coefficient_ad(system):
    transform = jnp.asarray(np.random.default_rng(112).normal(size=(8, 3))) * .2
    d = jnp.array([[.8, .1, .2], [.3, .7, -.1], [.1, .2, .4]])
    s = jnp.ones((2, 3, 3)) * .001
    def action(t, screen):
        return ProjectedNativeDirectBasis(system.provider, t).get_jk(d, screen_density=screen, screening_threshold=0., workers=2)
    def dense(t):
        dm = jnp.einsum('pi,ij,qj->pq', t, d, t)
        values = (jnp.einsum('pqrs,rs->pq', jnp.asarray(system.eri[0.]), dm),
                  jnp.einsum('prqs,rs->pq', jnp.asarray(system.eri[0.]), dm))
        return tuple(t.T @ value @ t for value in values)
    compare(action(transform, s), dense(transform))
    loss = lambda fn, t: sum(jnp.sum(x*x) for x in fn(t))
    compare(jax.grad(lambda t: loss(lambda x: action(x, s), t))(transform), jax.grad(lambda t: loss(dense, t))(transform), 1e-9)
    compare(action(transform, s), action(transform, s*0.))


def test_rsh_uses_each_operator_mask_with_one_screen_reference(system):
    from gradscf.scf.rsh import _direct_rsh_jk
    d, s = system.density, system.screen
    full, long_range = oracle(system, s, 0.)(d), oracle(system, s, OMEGA, with_j=False, with_k=True)(d)
    expected = full[0], .19*full[1] + (.65-.19)*long_range[1]
    actual = _direct_rsh_jk(system.provider, d, (OMEGA, .65, .19), screen_density=s,
                            screening_threshold=CUTOFF, workers=2)
    compare(actual, expected)


def test_positive_cutoff_retains_translation_response_and_primitive_basis_guard(system):
    def moved(t):
        parameters = replace(system.parameters, centers=system.parameters.centers + t)
        return NativeDirectBasis(system.plan, parameters).get_jk(system.density,
            screen_density=system.screen, screening_threshold=CUTOFF)
    translated = jax.jit(lambda t: jax.jvp(moved, (t,), (jnp.ones_like(t),))[1])(jnp.zeros((4, 3)))
    compare(translated, tuple(jnp.zeros_like(x) for x in translated), 1e-10)
    def changed(t):
        parameters = replace(system.parameters, exponents=tuple(a*jnp.exp(t) for a in system.parameters.exponents))
        return NativeDirectBasis(system.plan, parameters).get_jk(system.density,
            screen_density=system.screen, screening_threshold=CUTOFF)
    with pytest.raises((NotImplementedError, RuntimeError), match='(?i)(basis|exponent|coefficient)'):
        jax.block_until_ready(jax.jit(lambda t: jax.jvp(changed, (t,), (jnp.ones_like(t),))[1])(jnp.asarray(0.)))


def test_opaque_provider_cannot_silently_ignore_screen_reference():
    class Old:
        def get_jk(self, d): return d, 2*d
    with pytest.raises(NotImplementedError, match='(?i)(screen|provider)'):
        _selected_jk(Old(), jnp.eye(3), screen_density=jnp.eye(3), screening_threshold=CUTOFF)


@pytest.mark.parametrize('cart', [True, False], ids=['cartesian', 'spherical'])
def test_single_density_private_workers_matches_canonical_oracle(cart):
    top, parameters = integrals.prepare_basis('H 0 0 0; H .2 .1 1.8',
        [[0, [1.1, 1.]], [2, [.7, 1.]]], unit='Bohr', cart=cart)
    plan = integrals.make_plan(top)
    dimensions = [count * ((l+1)*(l+2)//2 if cart else 2*l+1)
                  for l, count in zip(top.angular_momenta, top.contraction_counts)]
    offsets = np.concatenate(([0], np.cumsum(dimensions)))
    n = top.nao
    assert n == offsets[-1] == (14 if cart else 12)
    screen = np.zeros((2, n, n), dtype=complex)
    for batch, shell, value in ((0, 1, .06j), (1, 3, .04*np.exp(.37j))):
        a, b = offsets[shell:shell+2]
        screen[batch, a:b, a:b] = value / (b-a)**2
    screen[0, offsets[0], offsets[2]] = .03j
    context = SimpleNamespace(offsets=offsets, eri={0.: np.asarray(plan.evaluate('eri', parameters))})
    d = jnp.asarray(np.random.default_rng(20261106).normal(size=(n, n)))
    provider = NativeDirectBasis(plan, parameters)
    # One real payload gives nb=1; two complete J/K outputs fit the private quota.
    action = lambda x, reference: provider.get_jk(x, screen_density=reference,
        screening_threshold=CUTOFF, workers=2)
    compare(jax.jit(action)(d, jnp.asarray(screen)), oracle(context, screen, 0.)(d))


@pytest.mark.parametrize('screened', [False, True], ids=['basis-only', 'fixed-reference'])
def test_zero_payload_preserves_nonzero_density_jvp_vjp_and_fd(system, screened):
    zero = jnp.zeros_like(system.density[0, 0])
    direction = system.density[0, 0]
    screen, cutoff = (system.screen, CUTOFF) if screened else (None, 0.)
    tensor = jnp.asarray(system.eri[0.])
    expected = (oracle(system, screen, 0.) if screened else lambda d:
        (jnp.einsum('pqrs,rs->pq', tensor, d), jnp.einsum('prqs,rs->pq', tensor, d)))
    action = lambda d: system.provider.get_jk(d, screen_density=screen,
        screening_threshold=cutoff, workers=2)
    value, tangent = jax.jit(lambda d, v: jax.jvp(action, (d,), (v,)))(zero, direction)
    compare(value, (zero, zero), 0.)
    compare(tangent, expected(direction), 1e-9)
    assert all(np.linalg.norm(np.asarray(x)) > 1e-8 for x in tangent)
    rng = np.random.default_rng(810)
    weights = tuple(jnp.asarray(rng.normal(size=zero.shape)+1j*rng.normal(size=zero.shape)) for _ in value)
    pullback = jax.jit(lambda cot: jax.vjp(action, zero)[1](cot)[0])(weights)
    compare(pullback, jax.vjp(expected, zero)[1](weights)[0], 1e-9)
    assert np.linalg.norm(np.asarray(pullback)) > 1e-8
    for step in (1e-4, 3e-5):
        fd = jax.tree.map(lambda a, b: (a-b)/(2*step), action(step*direction), action(-step*direction))
        compare(tangent, fd, 1e-9)


@pytest.mark.parametrize('invalid', ['environment', 'density-shape', 'negative-cutoff',
                                    'nonfinite-cutoff', 'screen-reference', 'screen-shape'])
def test_zero_payload_still_rejects_invalid_native_inputs(system, invalid):
    from gradscf.integrals.backends.native.jk import direct_jk
    atoms, basis, env = system.plan.pack(system.parameters)
    n = system.top.nao
    density, screen, cutoff = jnp.zeros((n, n)), None, 0.
    if invalid == 'environment': env = env.at[0].set(jnp.nan)
    elif invalid == 'density-shape': density = jnp.zeros((n+1, n+1))
    elif invalid == 'negative-cutoff': cutoff = -1.
    elif invalid == 'nonfinite-cutoff': cutoff = float('nan')
    elif invalid == 'screen-reference': screen, cutoff = jnp.full((n, n), jnp.nan), CUTOFF
    else: screen, cutoff = jnp.zeros((n+1, n+1)), CUTOFF
    with pytest.raises((ValueError, RuntimeError), match='(?i)(finite|cutoff|reference|shape|screen_density)'):
        jax.block_until_ready(jax.jit(lambda d: direct_jk(atoms, basis, env, d,
            cart=True, cutoff=cutoff, workers=2, screen_density=screen))(density))


def test_eight_batch_private_cap_matches_native_and_workspace_estimate():
    import re
    from pathlib import Path
    from gradscf.integrals.backends.native import jk
    source = (Path(__file__).parents[2]/'src/gradscf/integrals/_native/csrc/direct_jk.cc').read_text()
    assert jk._DIRECT_JK_PRIVATE_MAX_BATCH == 8
    assert re.search(r'kPrivateBatchLimit\s*=\s*8', source)
    top = SimpleNamespace(angular_momenta=(0,)*4, primitive_counts=(1,)*4,
        contraction_counts=(1,)*4, nuclear_charges=(1,), cart=True)
    provider = SimpleNamespace(plan=SimpleNamespace(topology=top))
    quota = 8*1024**2
    admitted = jk.native_jk_workspace_bytes(provider, 4, 0., private_output_bytes=quota)
    excluded = jk.native_jk_workspace_bytes(provider, 4, 0., private_output_bytes=0)
    assert admitted-excluded == 4*2*8*(4**2)*8


@pytest.mark.parametrize('payload', ['real8', 'complex4'])
@pytest.mark.parametrize('workers', [1, 4])
def test_eight_real_batch_private_values_and_density_ad(system, payload, workers):
    n = system.top.nao
    rng = np.random.default_rng(814)
    shape = (8 if payload == 'real8' else 4, n, n)
    density, direction = rng.normal(size=(2, *shape))
    if payload == 'complex4':
        density, direction = density+1j*rng.normal(size=shape), direction+1j*rng.normal(size=shape)
    density, direction = jnp.asarray(density), jnp.asarray(direction)
    # One fixed-reference sample exercises the screened ABI/private strategy;
    # the remaining samples retain the original generic linear operator.
    screen = system.screen if payload == 'complex4' and workers == 4 else None
    tensor = jnp.asarray(system.eri[0.])
    expected = (oracle(system, screen, 0.) if screen is not None else lambda d:
        (jnp.einsum('pqrs,...rs->...pq', tensor, d), jnp.einsum('prqs,...rs->...pq', tensor, d)))
    action = lambda d: system.provider.get_jk(d, screen_density=screen, workers=workers,
        screening_threshold=CUTOFF if screen is not None else 0.)
    value, tangent = jax.jit(lambda d, v: jax.jvp(action, (d,), (v,)))(density, direction)
    compare(value, expected(density), 1e-9)
    compare(tangent, expected(direction), 1e-9)
    weights = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in value)
    if payload == 'complex4':
        weights = tuple(w+1j*jnp.asarray(rng.normal(size=w.shape)) for w in weights)
    gradient = jax.jit(lambda cot: jax.vjp(action, density)[1](cot)[0])(weights)
    compare(gradient, jax.vjp(expected, density)[1](weights)[0], 1e-9)


@pytest.mark.skipif(os.environ.get('GRADSCF_EXPECT_DIAGNOSTICS') != '1',
                    reason='requires a separate --diagnostics library; counts are not clean timing')
def test_eight_real_batch_workers_evaluate_each_eri_quartet_once(system, record_property):
    from gradscf.integrals._native.diagnostics import get_counts, reset_counts
    # Four complex spin blocks become eight real FFI payload matrices.
    density = jnp.asarray(np.random.default_rng(815).normal(size=(4, system.top.nao, system.top.nao)), dtype=jnp.complex128)
    density = density+1j*jnp.sin(density.real)
    actions = {workers: jax.jit(lambda d, workers=workers: system.provider.get_jk(d, workers=workers))
               for workers in (1, 4)}
    for action in actions.values():
        jax.block_until_ready(action(density))
    values, counts = {}, {}
    for workers, action in actions.items():
        reset_counts()
        values[workers] = jax.block_until_ready(action(density))
        report = get_counts()
        counts[workers] = report['families']['eri4c']
        record_property('native_library_sha256', report['library_sha256'])
        for field, value in counts[workers].items():
            record_property(f'workers_{workers}_eri4c_{field}', value)
    compare(values[4], values[1], 1e-9)
    assert counts[1]['evaluations'] > 0
    # Worker-local scratch caches may issue different numbers of queries.
    assert counts[4]['evaluations'] == counts[1]['evaluations'], counts
