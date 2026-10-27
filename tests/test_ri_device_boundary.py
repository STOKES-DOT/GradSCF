"""Auxiliary RI is prepared on CPU before a resident GPU response reference.

Opt in with GRADSCF_TEST_GPU=1 and JAX_PLATFORMS=cuda,cpu. All comparisons
use float64, fixed geometry, untruncated auxiliary factors, and atomic units.
"""
from dataclasses import replace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def devices():
    if os.environ.get('GRADSCF_TEST_GPU') != '1':
        pytest.skip('Set GRADSCF_TEST_GPU=1 on an authorized visible GPU')
    return jax.devices('cpu')[0], jax.devices('gpu')[0]


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('preference', ['gpu', 'auto'])
def test_public_auxiliary_ri_prepares_cpu_and_reuses_gpu_reference(
        monkeypatch, devices, unrestricted, preference):
    from gradscf import gto, scf
    from gradscf.integrals.backends.native import density_fitting
    from gradscf.scf import builders, facade

    cpu, gpu = devices
    calls = {'ri': [], 'scf': 0, 'reference': 0, 'transfer': 0}
    original_ri = density_fitting.evaluate
    def ri(*args, **kwargs):
        assert jax.config.jax_default_device == cpu, 'Native RI preparation must select CPU'
        assert args[2].devices() == {cpu}, 'Packed RI input must reside on CPU'
        calls['ri'].append(kwargs['layout'])
        return original_ri(*args, **kwargs)
    monkeypatch.setattr(density_fitting, 'evaluate', ri)
    runner_name = 'run_uks_from_integrals' if unrestricted else 'run_rks_from_integrals'
    runner_owner = builders if unrestricted else facade
    original_run = getattr(runner_owner, runner_name)
    def run(**kwargs):
        calls['scf'] += 1
        assert jax.config.jax_default_device == cpu
        assert kwargs['df_factors'].devices() == {cpu}
        return original_run(**kwargs)
    monkeypatch.setattr(runner_owner, runner_name, run)
    reference_name = 'build_unrestricted_reference_from_facade' if unrestricted else 'build_restricted_reference_from_facade'
    original_reference = getattr(facade, reference_name)
    def build_reference(*args, **kwargs):
        calls['reference'] += 1
        assert jax.config.jax_default_device == cpu, 'Reference completion must select CPU'
        return original_reference(*args, **kwargs)
    monkeypatch.setattr(facade, reference_name, build_reference)
    from gradscf.tools import device
    original_put = device.put_molecule_on_device
    def transfer(*args, **kwargs):
        calls['transfer'] += 1
        return original_put(*args, **kwargs)
    monkeypatch.setattr(device, 'put_molecule_on_device', transfer)
    with jax.default_device(gpu):
        # Already committed input arrays exercise more than a default-device
        # context around newly created arrays. They must remain user-owned.
        coords = jax.device_put(jnp.array([0., 0., .8]), gpu)
        raw = jax.device_put(jnp.array([.8, .3]), gpu)
        mol = gto.M(atom=[('H', jnp.zeros(3)), ('H', coords)],
                    basis={'H': [[0, [1.2, raw[0]], [.3, raw[1]]],
                                 [0, [.15, 1.]]]}, spin=2 if unrestricted else 0)
        mf = (scf.UHF if unrestricted else scf.RHF)(mol,
            execution_device=preference, conv_tol=1e-11,
            conv_tol_density=1e-9, conv_tol_grad=1e-8).density_fit('weigend').run()
        assert mf.converged
        reference = mf.to_reference()
        for name in ('df_factors', 'mo_coeff', 'rdm1', 'h1e', 'dipole_integrals'):
            assert getattr(reference, name).devices() == {gpu}, name
        assert reference.grid.weights.devices() == {gpu}
        assert mf.to_reference() is reference
        assert calls == {'ri': [2, 3], 'scf': 1, 'reference': 1, 'transfer': 1}
        assert coords.devices() == raw.devices() == {gpu}
        mf.mol = replace(mol, atom='H 0 0 0; H 0 0 .9')
        with pytest.raises(RuntimeError, match='changed|kernel'):
            mf.to_reference()
        mf.run()
        assert mf.to_reference() is not reference
        assert calls == {'ri': [2, 3, 2, 3], 'scf': 2, 'reference': 2, 'transfer': 2}


def test_auxiliary_ri_cpu_products_survive_gpu_array_transfer(devices):
    from gradscf import integrals

    cpu, gpu = devices
    with jax.default_device(cpu):
        atom = 'H 0 0 0; H .2 .1 .8'
        topology, parameters = integrals.prepare_basis(atom,
            {'H': [[0, [1.1, .8], [.3, -.2]], [0, [.15, 1.]]]}, cart=True)
        auxiliary, auxiliary_parameters = integrals.prepare_basis(atom, 'weigend', cart=True)
        plan = integrals.make_auxiliary_plan(topology, auxiliary)
        metric = plan.metric_factor(parameters, auxiliary_parameters)
        coefficients, logalpha = parameters.coefficients, tuple(jnp.log(a) for a in parameters.exponents)
        values = coefficients, logalpha
        direction = jax.tree.map(lambda x: .03 * jnp.cos(jnp.arange(x.size).reshape(x.shape) + 1), values)
        def factors(values):
            c, a = values
            return plan.factors(replace(parameters, coefficients=c,
                exponents=tuple(jnp.exp(x) for x in a)), auxiliary_parameters, metric_factor=metric)
        primal, tangent = jax.jvp(factors, (values,), (direction,))
        cotangent = jnp.sin(jnp.arange(primal.size).reshape(primal.shape) + 1)
        gradient = jax.grad(lambda x: jnp.vdot(factors(x), cotangent))(values)
        step = 1e-4
        shift = lambda scale: jax.tree.map(lambda x, d: x + scale * d, values, direction)
        finite_difference = (factors(shift(step)) - factors(shift(-step))) / (2 * step)
        np.testing.assert_allclose(tangent, finite_difference, atol=3e-8, rtol=3e-7)
        lhs = sum(jnp.vdot(g, d) for g, d in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction)))
        np.testing.assert_allclose(lhs, jnp.vdot(cotangent, tangent), atol=3e-10, rtol=3e-10)
        with pytest.raises(NotImplementedError, match='geometry|auxiliary'):
            jax.jvp(lambda r: plan.factors(replace(parameters, centers=r),
                auxiliary_parameters, metric_factor=metric),
                (parameters.centers,), (jnp.ones_like(parameters.centers),))
    with jax.default_device(gpu):
        def moved_factors(x):
            # This is an eager CPU preparation followed by an explicit array
            # transfer, matching the production boundary rather than tracing
            # the CPU-only compact FFI into a CUDA executable.
            with jax.default_device(cpu):
                prepared = factors(x)
            return jax.device_put(prepared, gpu)
        moved_primal, moved_tangent = jax.jvp(moved_factors, (values,), (direction,))
        moved_gradient = jax.grad(lambda x: jnp.vdot(moved_factors(x),
            jax.device_put(cotangent, gpu)))(values)
        assert moved_primal.devices() == moved_tangent.devices() == {gpu}
        np.testing.assert_allclose(moved_primal, primal, atol=0., rtol=0.)
        np.testing.assert_allclose(moved_tangent, tangent, atol=3e-11, rtol=3e-11)
        for actual, expected in zip(jax.tree.leaves(moved_gradient), jax.tree.leaves(gradient)):
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
        for value in (primal, tangent, *jax.tree.leaves(gradient)):
            moved = jax.device_put(value, gpu)
            assert moved.devices() == {gpu}
            np.testing.assert_allclose(moved, value, atol=0., rtol=0.)
