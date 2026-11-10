"""Exact finite-molecule qsGW endpoint products and bounded working arrays."""
from importlib import import_module
from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.polarizability import rho_response_iw
from gradscf.gw.qsgw import _static_self_energy, qsgw_cd_restricted
from gradscf.gw.screened import screened_w_imag_axis_matrix
from gradscf.gw.self_energy import sigma_imag_matrix, sigma_residue_matrix
from gradscf.scf.autodiff import SCFDifferentiationConfig
from gradscf.solvers import LinearSolverConfig


def _inputs(eta=.02, nw=7):
    rng = np.random.default_rng(638)
    # Keep both ordered vertices independent. Both occupied and virtual
    # sectors contain an interior residue, and every endpoint has a half pole.
    factors = jnp.asarray(rng.normal(size=(5, 4, 4)) * .055)
    energy = jnp.array([-1.8, -.7, .25, 1.0])
    freqs, wts = scaled_legendre_grid(nw)
    return dict(b_mn=factors, b_ov=factors[:, :2, 2:], mo_energy=energy,
                nocc=2, ef=(energy[1] + energy[2]) / 2,
                freqs=freqs, wts=wts, eta=eta)


def _full_reference(kw):
    """Retain the old full-W definition as an independent contraction oracle."""
    energy, factors = kw['mo_energy'], kw['b_mn']
    response = lambda frequency: rho_response_iw(frequency, energy, kw['b_ov'])
    interaction = screened_w_imag_axis_matrix(factors, response, kw['freqs'])
    static = screened_w_imag_axis_matrix(factors, response, jnp.zeros(1))[0]
    arguments = dict(wmn_static=static,
                     occupation_sign=jnp.where(jnp.arange(energy.size) < kw['nocc'], -1., 1.))
    rows = []
    for p in range(energy.size):
        sigma = sigma_imag_matrix(energy[p], interaction, energy, kw['ef'],
                                 kw['freqs'], kw['wts'], kw['eta'], **arguments)
        sigma += sigma_residue_matrix(energy[p], energy, factors,
                                      ((energy, kw['b_ov'], 2.),), kw['ef'], kw['eta'], **arguments)
        rows.append((sigma[p] + sigma[:, p].conj()).real / 2)
    rows = jnp.stack(rows)
    return (rows + rows.T) / 2


def _with_arrays(kw, energy, factors):
    return {**kw, 'mo_energy': energy, 'b_mn': factors,
            'b_ov': factors[:, :2, 2:], 'ef': (energy[1] + energy[2]) / 2}


@pytest.mark.parametrize('eta', [0., .02])
def test_streaming_raw_endpoints_and_array_response_match_full_reference(eta):
    kw = _inputs(eta)
    inputs = (kw['mo_energy'], kw['b_mn'])
    rng = np.random.default_rng(639)
    direction = (jnp.array([.01, -.02, .03, -.01]),
                 jnp.asarray(rng.normal(size=inputs[1].shape) * .01))
    actual = lambda energy, factors: _static_self_energy(**_with_arrays(kw, energy, factors))
    expected = lambda energy, factors: _full_reference(_with_arrays(kw, energy, factors))
    primal, tangent = jax.jit(lambda args, tangents: jax.jvp(actual, args, tangents))(inputs, direction)
    reference, derivative = jax.jvp(expected, inputs, direction)
    assert np.isfinite(primal).all()  # eta=0 requires skipping inactive bare poles.
    np.testing.assert_allclose(primal, reference, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(primal, primal.T, atol=1e-14, rtol=0)
    np.testing.assert_allclose(tangent, derivative, atol=2e-11, rtol=1e-9)
    cotangent = jnp.array([[.1, .7, -.2, .4], [.7, -.3, .6, -.5],
                          [-.2, .6, .2, .8], [.4, -.5, .8, -.1]])
    loss = lambda energy, factors: jnp.sum(cotangent * actual(energy, factors))
    reference_loss = lambda energy, factors: jnp.sum(cotangent * expected(energy, factors))
    reverse = jax.jit(jax.grad(loss, argnums=(0, 1)))(*inputs)
    reference_reverse = jax.grad(reference_loss, argnums=(0, 1))(*inputs)
    for observed, target in zip(reverse, reference_reverse):
        np.testing.assert_allclose(observed, target, atol=2e-11, rtol=1e-9)
    inner = sum(jnp.vdot(dx, gradient) for dx, gradient in zip(direction, reverse))
    np.testing.assert_allclose(inner, jnp.sum(cotangent * tangent), atol=2e-11, rtol=1e-9)
    for step in (1e-4, 3e-5):
        plus = tuple(x + step * dx for x, dx in zip(inputs, direction))
        minus = tuple(x - step * dx for x, dx in zip(inputs, direction))
        finite_difference = (actual(*plus) - actual(*minus)) / (2 * step)
        np.testing.assert_allclose(tangent, finite_difference, atol=2e-10, rtol=2e-5)


def test_streaming_directional_second_response_matches_full_reference():
    kw = _inputs(nw=5)
    b = kw['b_mn'][:, :2, :2]
    def loss(t, reference=False):
        energy = jnp.array([-.7, .9])+t*jnp.array([.01, -.02])
        factors = b*(1+.07*t)
        arguments = {**kw, 'mo_energy': energy, 'b_mn': factors,
                     'b_ov': factors[:, :1, 1:], 'nocc': 1, 'ef': jnp.mean(energy)}
        potential = _full_reference(arguments) if reference else _static_self_energy(**arguments)
        return jnp.sum(jnp.array([[.1, .7], [.7, -.2]])*potential)
    curvature = jax.jit(lambda t: jax.jvp(jax.grad(loss), (t,), (jnp.ones_like(t),))[1])(0.)
    oracle = lambda t: loss(t, reference=True)
    expected = jax.jvp(jax.grad(oracle), (0.,), (1.,))[1]
    np.testing.assert_allclose(curvature, expected, atol=2e-11, rtol=1e-8)


def test_streaming_reuses_compilation_for_changed_same_shape_inputs(monkeypatch, record_property):
    module = import_module('gradscf.gw.qsgw')
    jax.clear_caches()
    original = module.rho_response_iw
    traces = []
    def observed(*args, **kwargs):
        traces.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'rho_response_iw', observed)
    first = _inputs()
    second = _with_arrays(first, first['mo_energy']+jnp.array([.01, -.02, .03, -.01]),
                          first['b_mn']*1.07)
    second.update(eta=.013, freqs=first['freqs']*1.03, wts=first['wts']*.98)
    actual_first = _static_self_energy(**first).block_until_ready()
    first_traces = len(traces)
    record_property('cold_screening_traces', first_traces)
    assert first_traces > 0, 'The cold call must trace the live screening response'
    actual_second = _static_self_energy(**second).block_until_ready()
    record_property('updated_inputs_screening_traces_total', len(traces))
    assert len(traces) == first_traces, 'Same-shape qsGW inputs retraced screening instead of reusing compilation'
    np.testing.assert_allclose(actual_first, _full_reference(first), atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(actual_second, _full_reference(second), atol=2e-12, rtol=1e-10)
    assert np.max(np.abs(np.asarray(actual_first-actual_second))) > 1e-8


def test_streaming_shares_one_retarded_zero_solve_for_all_own_poles(monkeypatch):
    module = import_module('gradscf.gw.qsgw')
    original = module._response_solve_one
    probes = []
    def observed(frequency, *args, **kwargs):
        jax.debug.callback(lambda value: probes.append(float(value)), frequency, ordered=True)
        return original(frequency, *args, **kwargs)
    monkeypatch.setattr(module, '_response_solve_one', observed)
    kw = _inputs(nw=5)
    factors, energy = kw['b_mn'][:, :2, :2], jnp.array([-.7, .9])
    kw.update(b_mn=factors, b_ov=factors[:, :1, 1:], mo_energy=energy,
              nocc=1, ef=jnp.mean(energy))
    jax.clear_caches()
    try:
        actual = _static_self_energy(**kw).block_until_ready()
        jax.effects_barrier()
        assert len(probes) == 1, f'Own poles repeated the retarded zero solve: {probes}'
        np.testing.assert_allclose(probes, [0.], atol=0, rtol=0)
        np.testing.assert_allclose(actual, _full_reference(kw), atol=2e-12, rtol=1e-10)
    finally:
        # Do not leave a callback executable in any later numerical benchmark.
        jax.clear_caches()


def _array_shapes(value):
    if hasattr(value, 'jaxpr'):
        yield from _array_shapes(value.jaxpr)
    elif hasattr(value, 'eqns'):
        for equation in value.eqns:
            for variable in equation.outvars:
                yield getattr(getattr(variable, 'aval', None), 'shape', ())
            yield from _array_shapes(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _array_shapes(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _array_shapes(child)


@pytest.mark.parametrize('reverse', [False, True])
def test_streaming_never_materializes_full_w_or_stacked_pole_dielectrics(reverse):
    kw = _inputs()
    naux, nmo, _ = kw['b_mn'].shape
    nw = len(kw['freqs'])
    def objective(factors):
        matrix = _static_self_energy(**_with_arrays(kw, kw['mo_energy'], factors))
        return jnp.sum(matrix ** 2)
    function = jax.grad(objective) if reverse else objective
    shapes = set(_array_shapes(jax.make_jaxpr(function)(kw['b_mn'])))
    forbidden = {(nw, nmo, nmo, nmo), (nmo, nmo, nmo),
                 (nmo, naux, naux), (nmo, nmo, naux, naux)}
    assert not shapes.intersection(forbidden), (
        f'qsGW retained full W or pole dielectric arrays: {shapes.intersection(forbidden)}')


def test_qsgw_plan_retains_no_frequency_or_endpoint_cubes():
    planner = getattr(import_module('gradscf.gw._memory'), 'plan_qsgw', None)
    assert callable(planner), 'qsGW needs a streaming working-array plan'
    b = np.zeros((17, 9, 9))
    ov = b[:, :2, 2:]
    large = planner(b, ov, nw=7, max_memory=16.)
    # Remove one byte from the roomy plan's exact allowance: at least one
    # RHS column must be tiled while preserving all physical orbitals.
    budget = (large.resident_bytes + large.working_bytes - 1) / 800_000
    small = planner(b, ov, nw=7, max_memory=budget)
    long_grid = planner(b, ov, nw=127, max_memory=16.)
    assert 1 <= small.rhs_batch < large.rhs_batch <= b.shape[1]
    assert small.frequency_batch == large.frequency_batch == 1
    assert small.resident_bytes + small.working_bytes <= int(budget * 800_000)
    assert long_grid.resident_bytes <= large.resident_bytes + (127 - 7) * 16
    with pytest.raises(ValueError, match='budget|memory'):
        planner(b, ov, nw=7, max_memory=1e-6)


def test_static_self_energy_budget_and_gmres_preserve_full_space():
    assert 'max_memory' in signature(_static_self_energy).parameters, 'Static qsGW must accept its caller budget'
    kw = _inputs()
    reference = _full_reference(kw)
    with pytest.raises(ValueError, match='budget|memory'):
        _static_self_energy(**kw, max_memory=1e-6)
    direct = _static_self_energy(**kw, max_memory=16., reserved_bytes=256)
    iterative = _static_self_energy(**kw, max_memory=16., reserved_bytes=256,
        screening_config=LinearSolverConfig(method='gmres', rtol=1e-11, atol=1e-13,
                                             maxiter=30, restart=10))
    np.testing.assert_allclose(direct, reference, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(iterative, reference, atol=2e-11, rtol=1e-9)


@pytest.mark.parametrize('implicit', [False, True])
def test_qsgw_driver_rejects_impossible_working_budget(implicit):
    assert 'max_memory' in signature(qsgw_cd_restricted).parameters, 'qsGW must accept its caller budget'
    differentiation = SCFDifferentiationConfig() if implicit else None
    with pytest.raises(ValueError, match='budget|memory'):
        qsgw_cd_restricted(mo_energy=jnp.array([-.6, .7]), mo_coeff=jnp.eye(2),
            nocc=1, df_factors=jnp.zeros((1, 2, 2)), hcore_matrix=jnp.diag(jnp.array([-.6, .7])),
            nw=5, max_memory=1e-6, differentiation=differentiation)


def test_qsgw_implicit_attachment_receives_mapping_budget(monkeypatch):
    assert 'max_memory' in signature(qsgw_cd_restricted).parameters, 'qsGW must accept its caller budget'
    module = import_module('gradscf.gw.qsgw')
    original = module.attach_root
    captured = []
    def observe(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'attach_root', observe)
    result = qsgw_cd_restricted(mo_energy=jnp.array([-.6, .7]), mo_coeff=jnp.eye(2),
        nocc=1, df_factors=jnp.zeros((1, 2, 2)), hcore_matrix=jnp.diag(jnp.array([-.6, .7])),
        nw=5, max_iter=3, max_memory=16., reserved_bytes=256,
        differentiation=SCFDifferentiationConfig(tolerance=1e-11))
    np.testing.assert_allclose(result.mo_energy, [-.6, .7], atol=1e-13, rtol=0)
    assert len(captured) == 1
    assert captured[0]['max_memory'] == 16.
    assert captured[0]['reserved_bytes'] >= 256
    assert captured[0]['operator_min_bytes'] > 0
